"""Reshapes whisper-server's verbose_json into the shape OpenAI's whisper-1 returns.

whisper-server (whisper.cpp's bundled HTTP server) already reports segments with avg_logprob and
no_speech_prob, but its per-segment "words" are really BPE tokens (" dishon" + "or", "!" on its
own), and it has no top-level words, usage or compression_ratio. Everything here is pure, so it is
unit tested against captured whisper-server output in tests/fixtures.
"""

import math
import re
import statistics
import unicodedata
import zlib

GRANULARITIES = ("word", "segment")

# Shortest a word may last once timings have been made monotonic.
MIN_WORD_SECONDS = 0.02

# Words ending a sentence are trimmed to twice the median word duration, itself capped at this, as
# the reference implementation does. See _trim_sentence_ends.
_MEDIAN_WORD_CAP_SECONDS = 0.7
_SENTENCE_END_MARKS = ".!?。！？"

# The reference implementation reports seek in mel frames (100 per second), one decode window
# (30 seconds) at a time.
_WINDOW_SECONDS = 30
_FRAMES_PER_SECOND = 100

# A segment that is nothing but an annotation, eg " [BLANK_AUDIO]", " [MUSIC PLAYING]" or " (music)".
# suppress_nst removes nearly all of them; this catches any that slip through.
_ANNOTATION = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\))\s*$")


def to_openai_verbose(raw: dict, granularities: list[str], drop_annotations: bool = True) -> dict:
    """whisper-server verbose_json -> OpenAI verbose_json for the requested granularities.

    Words are timed from DTW (t_dtw) when whisper-server was started with --dtw, which is far more
    accurate than its token t0/t1 after non-speech: without DTW the first word of a clip that opens
    with music is stamped at the start of the clip rather than when it is spoken.
    """
    duration = float(raw.get("duration", 0.0))

    # Kept segments, each with the words merged from its tokens.
    kept = []
    for segment in raw.get("segments", []):
        if drop_annotations and _ANNOTATION.match(segment.get("text", "")):
            continue
        kept.append((segment, _words_of(segment)))

    all_words = [word for _, words in kept for word in words]
    _settle(all_words, duration)
    _trim_sentence_ends(all_words)

    segments = [_segment(i, segment, words) for i, (segment, words) in enumerate(kept)]
    result = {
        "task": raw.get("task", "transcribe"),
        "language": raw.get("language", "english"),
        "duration": duration,
        "text": "".join(segment.get("text", "") for segment, _ in kept).strip(),
    }
    if "word" in granularities:
        result["words"] = [
            {"word": w["word"], "start": _round(w["start"]), "end": _round(w["end"])}
            for _, words in kept for w in words
        ]
    # OpenAI's default granularity is segment.
    if not granularities or "segment" in granularities:
        result["segments"] = segments
    # whisper-1 bills, and reports usage, in whole seconds of audio.
    result["usage"] = {"type": "duration", "seconds": math.ceil(duration)}
    return result


def _words_of(segment: dict) -> list[dict]:
    """Merges a segment's BPE tokens into words, without punctuation, like OpenAI's words."""
    groups = []
    for token in segment.get("words", []):
        # A leading space starts a new word, anything else continues the current one.
        if not groups or token.get("word", "").startswith(" "):
            groups.append([token])
        else:
            groups[-1].append(token)

    # Pure punctuation groups (eg a quote mark with a leading space) are not words. Each word keeps
    # its raw text, punctuation included, so sentence ends can be found later.
    named = [(raw, text, group) for group in groups
             if (text := _strip_punctuation(raw := "".join(token.get("word", "") for token in group)))]

    segment_end = float(segment.get("end", 0.0))
    words = []
    for raw, text, group in named:
        # DTW gives each token one aligned time and no end, so a DTW word runs until the next word
        # starts (_settle fills that in, across segments) and the final word to its segment's end.
        dtw = group[0].get("t_dtw", -1) >= 0
        start = _start_of(group[0])
        end = math.inf if dtw else float(group[-1].get("end", start))
        words.append({"word": text, "raw": raw, "start": start, "end": end, "segment_end": segment_end})
    return words


def _start_of(token: dict) -> float:
    t_dtw = token.get("t_dtw", -1)
    return t_dtw / 100 if t_dtw >= 0 else float(token.get("start", 0.0))


def _strip_punctuation(text: str) -> str:
    """Trims whitespace and leading/trailing punctuation, keeping inner marks: "don't", "re-run"."""
    text = text.strip()
    start, end = 0, len(text)
    while start < end and unicodedata.category(text[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(text[end - 1]).startswith("P"):
        end -= 1
    return text[start:end]


def _settle(words: list[dict], duration: float) -> None:
    """Makes timings safe to consume: starts never go backwards, words never overlap the next one,
    every word lasts at least MIN_WORD_SECONDS and nothing passes the end of the audio."""
    limit = duration if duration > 0 else math.inf
    floor = 0.0
    for word in words:
        word["start"] = min(max(word["start"], floor), limit)
        floor = word["start"]
    for i, word in enumerate(words):
        end = word["end"]
        if i + 1 < len(words):
            end = min(end, words[i + 1]["start"])
        elif end == math.inf:
            end = word["segment_end"]
        word["end"] = min(max(end, word["start"] + MIN_WORD_SECONDS), limit)


def _trim_sentence_ends(words: list[dict]) -> None:
    """Shortens over-long words that end a sentence, mirroring the reference implementation's
    word-timestamp heuristic. DTW only gives each word an onset, so the pause after a sentence ends
    up inside its last word. Trimming it restores the gap after the sentence. Pauses inside a
    sentence stay inside the word before them, as they do with whisper-1."""
    durations = [w["end"] - w["start"] for w in words if w["end"] > w["start"]]
    if not durations:
        return
    longest = 2 * min(_MEDIAN_WORD_CAP_SECONDS, statistics.median(durations))
    for word in words:
        if word["end"] - word["start"] > longest and word["raw"].rstrip()[-1:] in _SENTENCE_END_MARKS:
            word["end"] = word["start"] + longest


def _segment(index: int, segment: dict, words: list[dict]) -> dict:
    text = segment.get("text", "")
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))
    if words:
        # Bound the segment by its words, so a music intro absorbed into the segment is not part of it.
        start, end = words[0]["start"], words[-1]["end"]
    return {
        "id": index,
        "seek": int(start // _WINDOW_SECONDS) * _WINDOW_SECONDS * _FRAMES_PER_SECOND,
        "start": _round(start),
        "end": _round(end),
        "text": text,
        "tokens": segment.get("tokens", []),
        "temperature": segment.get("temperature", 0.0),
        "avg_logprob": segment.get("avg_logprob", 0.0),
        "compression_ratio": _compression_ratio(text),
        "no_speech_prob": segment.get("no_speech_prob", 0.0),
    }


def _compression_ratio(text: str) -> float:
    """Same measure as the reference implementation: a high ratio means repetitive, likely hallucinated, text."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def _round(seconds: float) -> float:
    return round(seconds, 3)
