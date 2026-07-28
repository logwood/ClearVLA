#!/usr/bin/env bash
set -euo pipefail

cd /home/sen.wang/workspace/robotics/clear/clearvla_v101_check_both

diagnostics=runs/diagnostics/v98_zero_shuffle
checkpoint="$diagnostics/checkpoint_snapshot.pt"
if [[ ! -f "$checkpoint" ]]; then
  checkpoint=runs/v98_bs8_formal_20260722_193507/checkpoints/latest.pt
fi

mkdir -p "$diagnostics"
export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

/home/sen.wang/Software/anaconda3/envs/XVLA/bin/python3.10 \
  -m clearvla.cli.eval_v39_policy \
  --checkpoint "$checkpoint" \
  --data-root /data/liang.zhang/dataset/grab_pen_single/grab_pen_single \
  --decoded-image-cache-dir /home/sen.wang/workspace/robotics/clear/data/cache_336 \
  --condition-mode dinov2-cache \
  --dinov2-token-cache-dir /home/sen.wang/workspace/robotics/clear/data/dinov2_cache_336 \
  --cache-resize 336 336 \
  --cameras top wrist \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --dtype bf16 \
  --eval-inference-steps 5 \
  --max-val-batches 4 \
  --flow-address-intervention-batches 4 \
  --out-json "$diagnostics/smoke.json" \
  > "$diagnostics/smoke.log" 2>&1

/home/sen.wang/Software/anaconda3/envs/XVLA/bin/python3.10 - "$diagnostics/smoke.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
probe = payload["flow_address_intervention"]
print(json.dumps({
    "checkpoint": payload["run_identity"],
    "representation": probe["representation"],
    "modes": probe["modes"],
    "paired": {
        key: value
        for key, value in probe["paired"].items()
        if not key.startswith("per_sample_")
    },
}, indent=2))
PY
