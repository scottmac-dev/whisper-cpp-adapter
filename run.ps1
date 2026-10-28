# One command to a running server: venv, dependencies, whisper.cpp discovery, launch.
#   .\run.ps1 [-Server] [-Port 8000]
#   -Server  use the whisper-server backend, which adds verbose_json (see README "Backends")
param([int]$Port = 8000, [switch]$Server)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSCommandPath)

$backend = if ($Server) { "server" } elseif ($env:WHISPER_BACKEND) { $env:WHISPER_BACKEND } else { "cli" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is required - see README" }

if (-not (Test-Path .venv)) { py -m venv .venv }
& .\.venv\Scripts\pip.exe install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# whisper-cli: an existing WHISPER_CLI wins, then a local or sibling checkout, else build one.
if (-not $env:WHISPER_CLI) {
    $found = Get-ChildItem -Path .\whisper.cpp\build, ..\whisper.cpp\build -Filter whisper-cli.exe `
             -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $env:WHISPER_CLI = $found.FullName }
}
if (-not $env:WHISPER_CLI) {
    Write-Host "No whisper.cpp build found - running scripts\setup_whisper.ps1"
    & .\scripts\setup_whisper.ps1
    $env:WHISPER_CLI = (Get-ChildItem -Path .\whisper.cpp\build -Filter whisper-cli.exe -Recurse |
                        Select-Object -First 1).FullName
}

# Model: an existing WHISPER_MODEL wins, else the first real model beside the binary.
if (-not $env:WHISPER_MODEL) {
    $models = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $env:WHISPER_CLI))) "models"
    $model = Get-ChildItem -Path $models -Filter "ggml-*.bin" -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -notlike "for-tests*" } | Select-Object -First 1
    if ($model) { $env:WHISPER_MODEL = $model.FullName }
}
if (-not $env:WHISPER_MODEL) { throw "No model found - run .\scripts\setup_whisper.ps1" }

# whisper-server: an existing WHISPER_SERVER wins, else the one beside whisper-cli, built if missing.
if ($backend -eq "server") {
    if (-not $env:WHISPER_SERVER) {
        $env:WHISPER_SERVER = Join-Path (Split-Path -Parent $env:WHISPER_CLI) "whisper-server.exe"
    }
    if (-not (Test-Path $env:WHISPER_SERVER)) {
        $build = Split-Path -Parent $env:WHISPER_CLI
        while ($build -and -not (Test-Path (Join-Path $build "CMakeCache.txt"))) { $build = Split-Path -Parent $build }
        if (-not $build) { throw "whisper-server.exe not found at $env:WHISPER_SERVER, and no CMake build to make it in" }
        Write-Host "Building whisper-server in $build"
        & cmake --build $build --config Release --target whisper-server
        if ($LASTEXITCODE -ne 0) { throw "building whisper-server failed" }
    }
}

$env:WHISPER_BACKEND = $backend
Write-Host "backend: $backend"
Write-Host "cli:     $env:WHISPER_CLI"
if ($backend -eq "server") { Write-Host "server:  $env:WHISPER_SERVER" }
Write-Host "model:   $env:WHISPER_MODEL"
Write-Host "docs:    http://localhost:$Port/docs"
& .\.venv\Scripts\hypercorn.exe app:app --bind "127.0.0.1:$Port"
