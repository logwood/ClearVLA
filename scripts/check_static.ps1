$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $root '.uv-cache' }
$env:UV_TOOL_DIR = if ($env:UV_TOOL_DIR) { $env:UV_TOOL_DIR } else { Join-Path $root '.uv-tools' }
$env:UV_TOOL_BIN_DIR = if ($env:UV_TOOL_BIN_DIR) { $env:UV_TOOL_BIN_DIR } else { Join-Path $root '.uv-bin' }

if (Get-Command ruff -ErrorAction SilentlyContinue) {
  & ruff check clearvla tests
}
else {
  & uv tool run ruff check clearvla tests
}

if (Get-Command pyright -ErrorAction SilentlyContinue) {
  & pyright
}
else {
  & uv tool run pyright
}
