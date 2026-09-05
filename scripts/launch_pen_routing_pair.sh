#!/usr/bin/env bash
# Launch exactly one member of the authorized matched Pen pair.
set -euo pipefail
variant="${1:?coarse_context or private_reader}"
gpu="${2:?physical GPU index}"
mode="${3:?smoke or formal}"
case "$variant" in coarse_context|private_reader) ;; *) exit 2 ;; esac
case "$mode" in smoke|formal) ;; *) exit 2 ;; esac
[[ "$gpu" =~ ^[0-6]$ ]] || exit 2
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
git diff --quiet
git diff --cached --quiet
commit="$(git rev-parse HEAD)"
PYTHON_BIN="${PYTHON_BIN:-/home/sen.wang/Software/anaconda3/envs/XVLA/bin/python3.10}"
RUN_BASE="${RUN_BASE:-/data/senwang/data/pen_bspline_routing_20260906}"
LOG_BASE="${LOG_BASE:-/data/senwang/logs/pen_bspline_routing_20260906}"
run="$RUN_BASE/${variant}_${mode}"
log="$LOG_BASE/${variant}_${mode}.log"
mkdir -p "$RUN_BASE" "$LOG_BASE"
if [[ -e "$run" || -e "$log" ]]; then
  printf 'Refusing duplicate output: %s or %s\n' "$run" "$log" >&2
  exit 3
fi
if [[ "$mode" == formal ]]; then
  "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] is True; assert len(r["runs"])==2; assert all(x["git"]==sys.argv[2] and x["checkpoint_readback"] for x in r["runs"])' "$LOG_BASE/gates.json" "$commit"
fi
used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
[[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )) || {
  printf 'GPU %s is occupied (%s MiB); refusing contention\n' "$gpu" "$used" >&2
  exit 4
}
# mkdir is atomic; retain the lock as a duplicate-launch guard.
mkdir "$RUN_BASE/${variant}_${mode}.launch-lock"
args=(--config "configs/mainline/object_intent_dynamics_323_pen_bspine_arm_${variant}.json"
      --device cuda --dtype bf16 --batch-size 8 --num-workers 4 --output-dir "$run")
if [[ "$mode" == smoke ]]; then
  args+=(--smoke --max-train-batches 8 --max-val-batches 1 --num-workers 0)
fi
printf 'source=%s variant=%s mode=%s gpu=%s init=fresh\n' "$commit" "$variant" "$mode" "$gpu" > "$log"
nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON_BIN" -u -m clearvla.mainline.train "${args[@]}" >> "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$RUN_BASE/${variant}_${mode}.pid"
printf 'pid=%s gpu=%s variant=%s mode=%s log=%s run=%s commit=%s\n' \
  "$pid" "$gpu" "$variant" "$mode" "$log" "$run" "$commit"
