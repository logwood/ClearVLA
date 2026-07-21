#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-$ROOT_DIR/.uv-tools}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-$ROOT_DIR/.uv-bin}"

if command -v ruff >/dev/null 2>&1; then
  ruff check clearvla tests
else
  uv tool run ruff check clearvla tests
fi

if command -v pyright >/dev/null 2>&1; then
  pyright
else
  uv tool run pyright
fi
