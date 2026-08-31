# Checks a running server end to end with a real .webm upload.
#   .\scripts\smoke_test.ps1 [-Port 8000]
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$base = "http://localhost:$Port"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "--- health ---"
    Invoke-RestMethod "$base/health" | ConvertTo-Json -Compress

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

    Write-Host "`n`nOK"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
