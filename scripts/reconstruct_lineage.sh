#!/usr/bin/env bash
# =============================================================================
# Reconstruct repo lineage from per-run deployment copies.
#
# Turns each deployment copy into an EXACT-content commit on a dedicated
# branch, in chronological order, without ever touching your main worktree.
# Safe against the two classic mistakes:
#   - stale-file mixing: tracked source dirs are wiped before each copy-in,
#     so every commit is precisely that deployment's content;
#   - fake dates: commit timestamps are taken from the copy's newest source
#     file mtime, so the reconstructed history carries truthful dates.
#
# Usage (run on the machine that has both the repo and the copies):
#   bash scripts/reconstruct_lineage.sh <repo_root> <base_ref> <branch> \
#        <copy_dir> <tag> <message> [<copy_dir> <tag> <message> ...]
#
# Example for the v76 generation (base = v75 commit):
#   bash scripts/reconstruct_lineage.sh ~/clearvla c339a01 v76-lineage \
#        ~/deploys/v76a_run   v76a-exact "v76a: owned intent decoder (run v76a.log)" \
#        ~/deploys/v76fx_run  v76fx-exact "v76-fx: parallel branch block (run v76.log)" \
#        ~/clearvla           v76.1      "v76.1: normalized residual amplitude constitution"
#
# The last triplet may point at the current repo itself to commit the present
# working state as the newest lineage node.  If an intermediate copy is lost,
# synthesize it first per history_design/archive/v76_code_reconstruction.md
# (path B),
# then feed the synthesized directory here.
# =============================================================================
set -euo pipefail

if [ "$#" -lt 6 ] || [ $(( ($# - 3) % 3 )) -ne 0 ]; then
    echo "usage: $0 <repo_root> <base_ref> <branch> <copy_dir> <tag> <message> [...]" >&2
    exit 1
fi

REPO="$(cd "$1" && pwd)"; BASE="$2"; BRANCH="$3"; shift 3
SOURCE_DIRS=(clearvla scripts tests)

WT="$(mktemp -d /tmp/lineage.XXXXXX)"
cleanup() { git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true; }
trap cleanup EXIT

git -C "$REPO" worktree add "$WT" "$BASE" >/dev/null
git -C "$WT" checkout -b "$BRANCH" >/dev/null

while [ "$#" -ge 3 ]; do
    SRC="$(cd "$1" && pwd)"; TAG="$2"; MSG="$3"; shift 3
    if [ ! -d "$SRC/clearvla" ]; then
        echo "SKIP (no clearvla/ inside): $SRC" >&2
        continue
    fi
    # Truthful timestamp = newest python source in the copy.
    STAMP="$(find "$SRC/clearvla" -name '*.py' -printf '%T@\n' 2>/dev/null | sort -n | tail -1)"
    DATE="$(date -d "@${STAMP%.*}" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S')"
    # Wipe tracked source dirs, then copy in the deployment's content exactly.
    ( cd "$WT" && git rm -rq --ignore-unmatch "${SOURCE_DIRS[@]}" 2>/dev/null || true )
    for d in "${SOURCE_DIRS[@]}"; do
        if [ -d "$SRC/$d" ]; then
            rsync -a --exclude '__pycache__' --exclude '*.pyc' "$SRC/$d/" "$WT/$d/"
        fi
    done
    ( cd "$WT" \
      && git add -A \
      && GIT_AUTHOR_DATE="$DATE" GIT_COMMITTER_DATE="$DATE" \
         git commit -q -m "$MSG" --allow-empty \
      && git tag -f "$TAG" )
    echo "committed: $TAG  ($DATE)  <- $SRC"
done

echo
echo "branch '$BRANCH' now holds the reconstructed lineage:"
git -C "$WT" log --oneline "$BASE..$BRANCH"
echo
echo "Continue future work from its tip:  git checkout $BRANCH"
