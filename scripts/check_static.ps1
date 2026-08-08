$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $root '.uv-cache' }

& uv run --frozen --no-sync ruff check clearvla tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& uv run --frozen --no-sync pyright --threads 4
exit $LASTEXITCODE
