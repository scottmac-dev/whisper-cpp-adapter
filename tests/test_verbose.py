"""verbose.to_openai_verbose against captured whisper-server output (tests/fixtures) and small synthetic cases.

Fixtures were captured from whisper.cpp 1.9.4's whisper-server with ggml-base.en and suppress_nst:
  jfk_after_noise_*      whisper.cpp's samples/jfk.wav after 5 s of pink noise, with and without --dtw base.en
  silence_annotation     10 s of digital silence without suppress_nst: a lone [BLANK_AUDIO] segment
  silence_hallucination  the same silence with suppress_nst and DTW: a hallucinated " you"
  noise_suppressed       8 s of pink noise: no segments at all
"""

import json
import math
import string
import zlib
from pathlib import Path

import pytest

import verbose

FIXTURES = Path(__file__).parent / "fixtures"
BOTH = ["word", "segment"]
JFK_WORDS = ("And so my fellow Americans ask not what your country can do "
             "for you ask what you can do for your country").split()


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def token(text, start, end, t_dtw=-1):
    return {"word": text, "start": start, "end": end, "t_dtw": t_dtw, "probability": 0.9}


def raw_with(segments, duration=10.0):
    return {"task": "transcribe", "language": "english", "duration": duration, "text": "", "segments": segments}


def segment(text, start, end, tokens, nsp=0.01, alp=-0.2):
    return {"id": 0, "text": text, "start": start, "end": end, "tokens": [1, 2], "words": tokens,
            "temperature": 0.0, "avg_logprob": alp, "no_speech_prob": nsp}


# --- shape --------------------------------------------------------------------------------------

def test_shape_matches_whisper_1_verbose_json():
    result = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), BOTH)
    assert list(result) == ["task", "language", "duration", "text", "words", "segments", "usage"]
    assert result["task"] == "transcribe"
    assert result["language"] == "english"
    assert all(set(w) == {"word", "start", "end"} for w in result["words"])
    assert all(set(s) == {"id", "seek", "start", "end", "text", "tokens", "temperature", "avg_logprob",
                          "compression_ratio", "no_speech_prob"} for s in result["segments"])
    assert result["usage"] == {"type": "duration", "seconds": 16}


@pytest.mark.parametrize("granularities, words, segments", [
    (["word"], True, False),
    (["segment"], False, True),
    (BOTH, True, True),
    ([], False, True),  # OpenAI's default granularity is segment
])
def test_granularities_select_words_and_segments(granularities, words, segments):
    result = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), granularities)
    assert ("words" in result) is words
    assert ("segments" in result) is segments


def test_text_is_joined_segment_text_without_leading_space():
    result = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), BOTH)
    assert result["text"] == ("And so my fellow Americans ask not what your country can do "
                              "for you, ask what you can do for your country.")
    assert result["segments"][0]["text"].startswith(" ")  # segments keep whisper's leading space


@pytest.mark.parametrize("duration, seconds", [(16.0, 16), (8.47, 9), (0.0, 0)])
def test_usage_is_whole_seconds_rounded_up(duration, seconds):
    assert verbose.to_openai_verbose(raw_with([], duration), BOTH)["usage"] == {"type": "duration", "seconds": seconds}


# --- words --------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["jfk_after_noise_dtw", "jfk_after_noise_no_dtw"])
def test_words_have_no_punctuation_or_spaces(name):
    words = [w["word"] for w in verbose.to_openai_verbose(fixture(name), ["word"])["words"]]
    assert words == JFK_WORDS
    assert not any(w[0] in string.punctuation or w[-1] in string.punctuation or " " in w for w in words)


def test_subword_tokens_merge_and_inner_apostrophes_survive():
    tokens = [token(" dishon", 0.0, 0.3), token("or", 0.3, 0.5), token(",", 0.5, 0.5),
              token(" don", 0.6, 0.8), token("'t", 0.8, 0.9), token(" \"", 0.9, 0.9), token("Run", 1.0, 1.2),
              token("!\"", 1.2, 1.3)]
    result = verbose.to_openai_verbose(raw_with([segment(' dishonor, don\'t "Run!"', 0.0, 1.3, tokens)]), ["word"])
    assert [w["word"] for w in result["words"]] == ["dishonor", "don't", "Run"]


def test_dtw_times_first_word_when_spoken_not_at_clip_start():
    # The clip opens with 5 s of noise. Token t0 puts "And" at 0.00; DTW puts it where it is spoken.
    dtw = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), BOTH)
    no_dtw = verbose.to_openai_verbose(fixture("jfk_after_noise_no_dtw"), BOTH)
    assert dtw["words"][0]["start"] >= 5.0
    assert dtw["segments"][0]["start"] == dtw["words"][0]["start"]
    assert no_dtw["words"][0]["start"] == 0.0


@pytest.mark.parametrize("name", ["jfk_after_noise_dtw", "jfk_after_noise_no_dtw"])
def test_word_timings_are_monotonic_non_overlapping_and_in_range(name):
    result = verbose.to_openai_verbose(fixture(name), BOTH)
    words = result["words"]
    for word in words:
        assert 0 <= word["start"] < word["end"] <= result["duration"]
    for word, following in zip(words, words[1:]):
        assert word["start"] <= following["start"]
        assert word["end"] <= following["start"]


def test_dtw_words_run_to_the_next_word_across_segments():
    result = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), BOTH)
    words, segments = result["words"], result["segments"]
    # Without a following word to bound it, the last word of a segment ends at the next segment's start.
    last_of_first = [w for w in words if w["start"] < segments[1]["start"]][-1]
    assert last_of_first["end"] == segments[1]["start"]


def test_long_word_ending_a_sentence_is_trimmed():
    # DTW only gives onsets, so "Hello." would otherwise absorb the 5 s pause after it.
    tokens = [token(" Hi", 0.0, 0.0, 0), token(" there", 0.0, 0.0, 30), token(" said", 0.0, 0.0, 60),
              token(" Hello", 0.0, 0.0, 90), token(".", 0.0, 0.0, 110), token(" World", 0.0, 0.0, 600)]
    result = verbose.to_openai_verbose(raw_with([segment(" Hi there said Hello. World", 0.0, 6.5, tokens)]), ["word"])
    hello = next(w for w in result["words"] if w["word"] == "Hello")
    assert hello["start"] == 0.9
    assert hello["end"] == pytest.approx(0.9 + 2 * 0.3)  # twice the median word duration
    assert result["words"][-1]["start"] == 6.0


def test_long_word_inside_a_sentence_keeps_its_pause():
    # Matches whisper-1: only sentence ends are trimmed, so "Americans" keeps the pause after it.
    words = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), ["word"])["words"]
    americans = next(w for w in words if w["word"] == "Americans")
    assert americans["end"] - americans["start"] > 1.0


# --- segments -----------------------------------------------------------------------------------

def test_scores_pass_through_unchanged():
    raw = fixture("jfk_after_noise_dtw")
    result = verbose.to_openai_verbose(raw, BOTH)
    for ours, theirs in zip(result["segments"], raw["segments"]):
        assert ours["avg_logprob"] == theirs["avg_logprob"]
        assert ours["no_speech_prob"] == theirs["no_speech_prob"]
        assert ours["tokens"] == theirs["tokens"]


def test_compression_ratio_matches_the_reference_measure():
    result = verbose.to_openai_verbose(fixture("jfk_after_noise_dtw"), ["segment"])
    for seg in result["segments"]:
        data = seg["text"].encode()
        assert seg["compression_ratio"] == pytest.approx(len(data) / len(zlib.compress(data)))


def test_seek_is_the_start_of_the_30_second_window_in_frames():
    tokens = [token(" late", 31.0, 31.5)]
    result = verbose.to_openai_verbose(raw_with([segment(" late", 31.0, 31.5, tokens)], duration=40.0), ["segment"])
    assert result["segments"][0]["seek"] == 3000


# --- non-speech ---------------------------------------------------------------------------------

def test_annotation_segments_are_dropped():
    result = verbose.to_openai_verbose(fixture("silence_annotation"), BOTH)
    assert result["text"] == ""
    assert result["words"] == [] and result["segments"] == []
    assert result["usage"]["seconds"] == 10


def test_annotations_kept_when_dropping_is_off():
    result = verbose.to_openai_verbose(fixture("silence_annotation"), BOTH, drop_annotations=False)
    assert result["segments"][0]["text"].strip() == "[BLANK_AUDIO]"


def test_ids_are_renumbered_after_dropping():
    raw = raw_with([segment(" [MUSIC PLAYING]", 0.0, 2.0, [token(" [", 0.0, 0.1)]),
                    segment(" Hello.", 2.0, 3.0, [token(" Hello", 2.0, 2.5), token(".", 2.5, 3.0)])])
    result = verbose.to_openai_verbose(raw, BOTH)
    assert [(s["id"], s["text"]) for s in result["segments"]] == [(0, " Hello.")]


def test_no_speech_gives_empty_result():
    result = verbose.to_openai_verbose(fixture("noise_suppressed"), BOTH)
    assert (result["text"], result["words"], result["segments"]) == ("", [], [])


def test_hallucinations_are_passed_through_for_the_caller_to_filter():
    # whisper hallucinates " you" on silence. Its no_speech_prob is high but avg_logprob is not low,
    # so it survives the "no_speech_prob > 0.6 and avg_logprob < -1.0" rule. The adapter reports, it
    # does not filter, so callers see exactly what whisper-1 would give them.
    result = verbose.to_openai_verbose(fixture("silence_hallucination_dtw"), BOTH)
    (seg,) = result["segments"]
    assert seg["text"].strip() == "you"
    assert seg["no_speech_prob"] > 0.6 and seg["avg_logprob"] > -1.0
    assert not math.isinf(result["words"][0]["end"])
