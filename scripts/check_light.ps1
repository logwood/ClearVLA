$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$env:UV_CACHE_DIR = if ($env:UV_CACHE_DIR) { $env:UV_CACHE_DIR } else { Join-Path $root '.uv-cache' }

& uv run --frozen --no-sync python -m compileall -q clearvla tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$tests = @(
    'tests/test_audit_policy_logs.py'
    'tests/test_probe_flow_dino_dataset_motion.py'
    'tests/test_physical_action_codec.py'
    'tests/test_temporal_dct.py'
)

& uv run --frozen --no-sync python -m pytest -q @tests
exit $LASTEXITCODE
