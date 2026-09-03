#!/usr/bin/env bash
# Incrementally back up a run directory that is ALREADY IN PROGRESS.
#
# A run started before engine.sync existed (or with sync disabled) cannot grow
# the feature mid-flight, so this sidecar does the same job from outside: an
# rclone loop in its own process, which by construction cannot slow the
# evaluation down or crash it.
#
#     bash tools/sync_run.sh runs/20260903-184330_full            # foreground
#     nohup bash tools/sync_run.sh runs/20260903-184330_full \
#           > /tmp/abench_sync.log 2>&1 &                          # detached
#
# It keeps running until you stop it (Ctrl-C / kill), uploading only what
# changed each pass, and does one final pass on exit so the reports written at
# the very end are included.
#
# Environment:
#   SYNC_REMOTE    rclone destination        (default: gdrive:AbductionBench)
#   SYNC_INTERVAL  seconds between passes    (default: 60)
#   SYNC_EXCLUDE   comma-separated globs     (default: none; try "raw/**")
#   SYNC_BWLIMIT   e.g. 8M                   (default: unlimited)

set -uo pipefail   # deliberately not -e: a failed pass must not end the loop

RUN_DIR="${1:-}"
REMOTE="${SYNC_REMOTE:-gdrive:AbductionBench}"
INTERVAL="${SYNC_INTERVAL:-60}"
BWLIMIT="${SYNC_BWLIMIT:-}"
EXCLUDES="${SYNC_EXCLUDE:-}"

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "usage: bash tools/sync_run.sh <run-dir>   (e.g. runs/20260903-184330_full)" >&2
  exit 2
fi
if ! command -v rclone >/dev/null; then
  echo "rclone is not installed" >&2
  exit 1
fi

RUN_ID="$(basename "$RUN_DIR")"
DEST="${REMOTE%/}/$RUN_ID"

args=(copy "$RUN_DIR" "$DEST" --update --transfers=4 --checkers=8
      --timeout=300s --retries=2 --low-level-retries=3 --fast-list --stats=0)
[[ -n "$BWLIMIT" ]] && args+=("--bwlimit=$BWLIMIT")
if [[ -n "$EXCLUDES" ]]; then
  IFS=',' read -ra patterns <<< "$EXCLUDES"
  for pattern in "${patterns[@]}"; do args+=(--exclude "$pattern"); done
fi

# Verify the destination is usable before claiming to back anything up.
remote_name="${REMOTE%%:*}"
if [[ "$REMOTE" == *:* ]] && ! rclone listremotes 2>/dev/null | grep -qx "$remote_name:"; then
  echo "rclone remote '$remote_name' is not configured." >&2
  echo "Run:  bash tools/setup_drive_remote.sh '<token-json>'" >&2
  exit 1
fi

final_pass() {
  echo "[$(date -Is)] final pass"
  rclone "${args[@]}"
  echo "[$(date -Is)] stopped"
  exit 0
}
trap final_pass INT TERM

echo "[$(date -Is)] backing up $RUN_DIR -> $DEST every ${INTERVAL}s (incremental)"
passes=0
failures=0
while true; do
  start=$SECONDS
  if rclone "${args[@]}"; then
    passes=$((passes + 1))
    echo "[$(date -Is)] pass $passes ok in $((SECONDS - start))s ($failures failure(s) so far)"
  else
    failures=$((failures + 1))
    echo "[$(date -Is)] pass FAILED (${failures} total) -- retrying in ${INTERVAL}s" >&2
  fi
  sleep "$INTERVAL"
done
