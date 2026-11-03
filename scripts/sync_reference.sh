#!/usr/bin/env bash
# Refresh reference/ from a local road-to-robotics clone (the snapshot goes stale as the Orin keeps working).
#   bash scripts/sync_reference.sh ~/src/road-to-robotics
# Copies the book, the paper folders and the register files; never TIMELINE.md or strategy/.
set -euo pipefail
SRC="${1:?usage: sync_reference.sh <path to road-to-robotics>}"
DST="$(cd "$(dirname "$0")/.." && pwd)/reference/journal"
git -C "$SRC" pull --rebase --quiet || echo "warning: could not pull $SRC, using it as is"
mkdir -p "$DST"
rm -rf "$DST/book" "$DST/papers"
cp -r "$SRC/journal/book" "$SRC/journal/papers" "$DST/"
cp "$SRC/journal/EXPERIMENTS.md" "$SRC/journal/LESSONS.md" "$SRC/journal/DECISIONS.md" "$SRC/journal/METHOD.md" "$DST/"
COMMIT="$(git -C "$SRC" rev-parse --short HEAD)"
echo "road-to-robotics @ $COMMIT, synced $(date -u +%Y-%m-%dT%H:%MZ)" > "$(dirname "$DST")/SNAPSHOT.txt"
echo "reference/ synced from $COMMIT. Re-run W8 D4 tests: the numbers you cite may have moved."
