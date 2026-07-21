#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_policy_golden_worktrees.sh BASELINE_REF CANDIDATE_REF VARIANT OUTPUT_DIR [PYTHON]

Examples:
  bash scripts/run_policy_golden_worktrees.sh \
    <fixed-v77-baseline-ref> HEAD v77 runs/golden/v77

  bash scripts/run_policy_golden_worktrees.sh \
    v76-owned-intent-mmdit-checkpoint my-refactor-tag v76 runs/golden/v76

VARIANT is v76, v77, v78, v79, v80, v81, v82, v84, or v88. Both refs are captured in isolated processes and
detached temporary worktrees. The script exits nonzero on any difference.
Use a baseline that has already passed the v2 health capture; historical v1
refs may be intentionally rejected for optimizer ownership or finite-value
violations discovered by the stricter harness.
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

BASELINE_REF=$1
CANDIDATE_REF=$2
VARIANT=$3
OUTPUT_DIR=$4
PYTHON_BIN=${5:-python}

case $VARIANT in
  v76|v77|v78|v79|v80|v81|v82|v84|v88) ;;
  *)
    echo "VARIANT must be v76, v77, v78, v79, v80, v81, v82, v84, or v88" >&2
    exit 2
    ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
HARNESS=$REPO_ROOT/clearvla/tools/policy_golden.py
OUTPUT_DIR=$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/clearvla-policy-golden.XXXXXX")
BASELINE_WORKTREE=$TMP_ROOT/baseline
CANDIDATE_WORKTREE=$TMP_ROOT/candidate
SUPPORT_TREES=(clearvla/data)

cleanup() {
  git -C "$REPO_ROOT" worktree remove --force "$BASELINE_WORKTREE" >/dev/null 2>&1 || true
  git -C "$REPO_ROOT" worktree remove --force "$CANDIDATE_WORKTREE" >/dev/null 2>&1 || true
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add --detach "$BASELINE_WORKTREE" "$BASELINE_REF"
git -C "$REPO_ROOT" worktree add --detach "$CANDIDATE_WORKTREE" "$CANDIDATE_REF"

ensure_support_trees() {
  local worktree=$1
  local relative source target
  for relative in "${SUPPORT_TREES[@]}"; do
    source=$REPO_ROOT/$relative
    target=$worktree/$relative
    if [[ -d $target ]]; then
      echo "support tree from ref: $relative"
      continue
    fi
    if [[ -e $target ]]; then
      echo "support tree target exists but is not a directory: $target" >&2
      exit 1
    fi
    if [[ ! -d $source ]]; then
      echo "required support tree is absent from ref and checkout: $relative" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$target")"
    ln -s "$source" "$target"
    echo "support tree overlay: $relative <- $source"
  done
}

ensure_support_trees "$BASELINE_WORKTREE"
ensure_support_trees "$CANDIDATE_WORKTREE"

BASELINE_SUPPORT_ARGS=()
CANDIDATE_SUPPORT_ARGS=()
for relative in "${SUPPORT_TREES[@]}"; do
  BASELINE_SUPPORT_ARGS+=(--support-tree "$relative=$BASELINE_WORKTREE/$relative")
  CANDIDATE_SUPPORT_ARGS+=(--support-tree "$relative=$CANDIDATE_WORKTREE/$relative")
done

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" fixture \
  --output "$OUTPUT_DIR/fixture.pt"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" capture \
  --source-root "$BASELINE_WORKTREE" \
  --fixture "$OUTPUT_DIR/fixture.pt" \
  --variant "$VARIANT" \
  "${BASELINE_SUPPORT_ARGS[@]}" \
  --output "$OUTPUT_DIR/baseline"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" capture \
  --source-root "$CANDIDATE_WORKTREE" \
  --fixture "$OUTPUT_DIR/fixture.pt" \
  --variant "$VARIANT" \
  "${CANDIDATE_SUPPORT_ARGS[@]}" \
  --output "$OUTPUT_DIR/candidate"

PYTHONHASHSEED=0 "$PYTHON_BIN" "$HARNESS" compare \
  --baseline "$OUTPUT_DIR/baseline" \
  --candidate "$OUTPUT_DIR/candidate" \
  --report "$OUTPUT_DIR/comparison.json"

echo "golden comparison passed: $OUTPUT_DIR/comparison.json"
