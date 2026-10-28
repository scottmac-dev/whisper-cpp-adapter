# Clones and builds whisper.cpp on Windows, then downloads a model.
# Skip this entirely if you already have whisper.cpp: just set WHISPER_CLI and WHISPER_MODEL.
#
#   .\scripts\setup_whisper.ps1              # base.en (~150 MB)
#   .\scripts\setup_whisper.ps1 -Model tiny.en
#
# Needs git, cmake and ffmpeg on PATH, plus Visual Studio Build Tools (Desktop C++ workload).
param([string]$Model = "base.en")

$ErrorActionPreference = "Stop"

# $ErrorActionPreference does not apply to native programs, so check exit codes by hand.
function Invoke-Checked {
    param([Parameter(Mandatory)][string]$Exe,
          [Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

$hints = @{ git = "Git.Git"; cmake = "Kitware.CMake"; ffmpeg = "Gyan.FFmpeg" }
foreach ($tool in $hints.Keys) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool was not found on PATH. Install it with: winget install $($hints[$tool])"
    }
}

$Dir = Join-Path (Split-Path -Parent $PSScriptRoot) "whisper.cpp"

if (-not (Test-Path $Dir)) {
    Invoke-Checked git clone --depth 1 https://github.com/ggml-org/whisper.cpp $Dir
}

Invoke-Checked cmake -B "$Dir\build" -S "$Dir" -DCMAKE_BUILD_TYPE=Release
Invoke-Checked cmake --build "$Dir\build" --config Release --target whisper-cli whisper-server

Invoke-Checked "$Dir\models\download-ggml-model.cmd" $Model "$Dir\models"

# MSVC drops the binary in build\bin\Release; other generators use build\bin.
$cli = Get-ChildItem -Path "$Dir\build" -Filter whisper-cli.exe -Recurse | Select-Object -First 1
if (-not $cli) { throw "Build finished but whisper-cli.exe was not found under $Dir\build" }

Write-Host ""
Write-Host "Done. Run the server with:"
Write-Host "  `$env:WHISPER_CLI = `"$($cli.FullName)`""
Write-Host "  `$env:WHISPER_MODEL = `"$Dir\models\ggml-$Model.bin`""
Write-Host "  hypercorn app:app --bind 127.0.0.1:8000"
Write-Host ""
Write-Host "Or just .\run.ps1 (add -Server for verbose_json, see README \"Backends\")."
