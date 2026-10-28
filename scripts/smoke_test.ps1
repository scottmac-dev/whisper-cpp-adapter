# Checks a running server end to end with a real .webm upload, plus the checks for its backend.
#   .\scripts\smoke_test.ps1 [-Port 8000]
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$base = "http://localhost:$Port"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "--- health ---"
    $health = Invoke-RestMethod "$base/health"
    $health | ConvertTo-Json -Compress
    $backend = if ($health.backend) { $health.backend } else { "cli" }

    $src = @(".\whisper.cpp\samples\jfk.wav", "..\whisper.cpp\samples\jfk.wav") |
           Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $src) { throw "no sample audio found in whisper.cpp\samples" }

    $webm = Join-Path $tmp "sample.webm"
    ffmpeg -nostdin -loglevel error -y -i $src -c:a libopus $webm
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed" }

    # curl.exe ships with Windows 10+ and builds the multipart body correctly.
    Write-Host "`n--- response_format=json ---"
    curl.exe -sf -X POST "$base/v1/audio/transcriptions" -H "Authorization: Bearer sk-not-needed" `
      -F "file=@$webm" -F "model=gpt-4o-mini-transcribe" -F "response_format=json"
    if ($LASTEXITCODE -ne 0) { throw "json request failed" }

    Write-Host "`n`n--- response_format=text ---"
    curl.exe -sf -X POST "$base/v1/audio/transcriptions" -F "file=@$webm" -F "response_format=text"
    if ($LASTEXITCODE -ne 0) { throw "text request failed" }

    $body = Join-Path $tmp "body.json"
    if ($backend -eq "cli") {
        Write-Host "`n`n--- response_format=verbose_json is refused by the cli backend ---"
        $code = curl.exe -s -o $body -w "%{http_code}" -X POST "$base/v1/audio/transcriptions" `
          -F "file=@$webm" -F "response_format=verbose_json"
        Get-Content $body
        if ($code -ne "400") { throw "expected 400, got $code" }
    } else {
        # 5 s of noise before the sample: a word timed from the clip's start instead of when it is
        # spoken means DTW is not working.
        $late = Join-Path $tmp "late.wav"
        ffmpeg -nostdin -loglevel error -y -f lavfi -i "anoisesrc=d=5:c=pink:a=0.05:seed=1" -i $src `
          -filter_complex "[0:a]aresample=16000,aformat=channel_layouts=mono[n];[1:a]aresample=16000,aformat=channel_layouts=mono[s];[n][s]concat=n=2:v=0:a=1" `
          $late
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed" }

        Write-Host "`n`n--- response_format=verbose_json with word and segment timestamps, as an OpenAI whisper-1 client sends ---"
        curl.exe -sf -o $body -X POST "$base/v1/audio/transcriptions" -H "Authorization: Bearer sk-not-needed" `
          -F "file=@$late" -F "model=whisper-1" -F "response_format=verbose_json" `
          -F "timestamp_granularities[]=word" -F "timestamp_granularities[]=segment"
        if ($LASTEXITCODE -ne 0) { throw "verbose_json request failed" }
        $verbose = Get-Content $body -Raw | ConvertFrom-Json
        foreach ($key in "task", "language", "duration", "text", "words", "segments", "usage") {
            if (-not ($verbose.PSObject.Properties.Name -contains $key)) { throw "missing $key" }
        }
        if (-not $verbose.words -or -not $verbose.segments) { throw "no words or segments" }
        if ($verbose.usage.type -ne "duration" -or $verbose.usage.seconds -ne [math]::Ceiling($verbose.duration)) {
            throw "unexpected usage: $($verbose.usage | ConvertTo-Json -Compress)"
        }
        $first = $verbose.words[0]
        Write-Host "$($verbose.text)"
        Write-Host "$($verbose.words.Count) words, $($verbose.segments.Count) segments, first word $($first | ConvertTo-Json -Compress)"
        if (-not $health.dtw) {
            Write-Host "DTW is off, skipping the word timing check"
        } elseif ($first.start -lt 4.5) {
            throw "first word at $($first.start)s, but speech starts after 5 s of noise"
        }

        Write-Host "`n--- timestamp_granularities[] without verbose_json is refused ---"
        $code = curl.exe -s -o $body -w "%{http_code}" -X POST "$base/v1/audio/transcriptions" `
          -F "file=@$webm" -F "response_format=json" -F "timestamp_granularities[]=word"
        Get-Content $body
        if ($code -ne "400") { throw "expected 400, got $code" }
    }

    Write-Host "`n`nOK ($backend backend)"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
