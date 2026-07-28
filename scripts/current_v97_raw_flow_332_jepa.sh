#!/usr/bin/env bash
# V97 is an archived experiment label. The current source implements the V98
# DINO-seeded contract, so this entry point forwards explicitly instead of
# producing a second, incompatible run under the old V97 name.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
printf '[v97] archived_contract=raw_global_correlation current_contract=v98 forwarding=1\n' >&2
exec bash "${SCRIPT_DIR}/current_v98_dino_seeded_raw_flow_332_jepa.sh" "$@"
