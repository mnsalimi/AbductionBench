#!/usr/bin/env bash
# Incrementally back up a run directory that is ALREADY IN PROGRESS.
#
# A run started before engine.sync existed (or with sync disabled) cannot grow
# the feature mid-flight, so this sidecar does the same job from outside: an
# rclone loop in its own process, which by construction cannot slow the
# evaluation down or crash it.
#
#     bash tools/sync_run.sh runs/20260903-190409_full            # foreground
#     nohup bash tools/sync_run.sh runs/20260903-190409_full \
#           > /tmp/abench_sync.log 2>&1 &                          # detached
#
# Two things it does that a naive "rclone copy" does not, both learned from a
# live run against Google Drive:
#
#   * It uploads a SNAPSHOT, not the live directory. records.jsonl and the logs
#     are appended to continuously, so rclone would size/hash a file, upload it,
#     find the remote copy no longer matches, call the transfer corrupt and
#     DELETE it from the remote -- losing exactly the files worth keeping. An
#     rsync into a staging directory takes a moment and makes every upload a
#     stable, verifiable file.
#   * It excludes raw/ by default and throttles API calls. In a full run 1,096
#     of 1,201 files are per-batch debug payloads; uploading them exhausts
#     Google Drive's per-minute request quota for rclone's shared OAuth client
#     (HTTP 403 rateLimitExceeded) and starves the files that matter. Every
#     record, metric, checkpoint, log and report is still backed up.
#
# It runs until stopped (Ctrl-C / kill) and does one final pass on exit, so the
# reports written at the very end are included. The final pass then VERIFIES the
# remote and re-sends anything missing, because a failed transfer still leaves
# the destination directory behind: without the check, a run can end with a
# complete Excel report next to empty task directories.
#
# Environment:
#   SYNC_REMOTE    rclone destination        (default: gdrive:AbductionBench)
#   SYNC_INTERVAL  seconds between passes    (default: 60)
#   SYNC_EXCLUDE   comma-separated globs     (default: "raw/"; set to "" for all)
#   SYNC_TPSLIMIT  API calls per second      (default: 8; Drive's shared client is strict)
#   SYNC_BWLIMIT   e.g. 8M                   (default: unlimited)
#   SYNC_STAGE     staging directory         (default: $TMPDIR/abench_sync_stage)

set -uo pipefail   # deliberately not -e: a failed pass must not end the loop

RUN_DIR="${1:-}"
REMOTE="${SYNC_REMOTE:-gdrive:AbductionBench}"
INTERVAL="${SYNC_INTERVAL:-60}"
BWLIMIT="${SYNC_BWLIMIT:-}"
TPSLIMIT="${SYNC_TPSLIMIT:-8}"
EXCLUDES="${SYNC_EXCLUDE-raw/}"
STAGE_ROOT="${SYNC_STAGE:-${TMPDIR:-/tmp}/abench_sync_stage}"

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "usage: bash tools/sync_run.sh <run-dir>   (e.g. runs/20260903-190409_full)" >&2
  exit 2
fi
command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 1; }
command -v rsync  >/dev/null || { echo "rsync is not installed" >&2; exit 1; }

RUN_ID="$(basename "$(cd "$RUN_DIR" && pwd)")"
DEST="${REMOTE%/}/$RUN_ID"
STAGE="$STAGE_ROOT/$RUN_ID"
mkdir -p "$STAGE"

# One sidecar per run, enforced. Two of them upload the same files at the same
# time, and Google Drive happily stores same-named duplicates -- which then need
# an `rclone dedupe` to clean up. flock releases automatically when this process
# dies, so a crashed sidecar does not block the next one.
# The lock lives OUTSIDE the staged tree: rsync --delete would otherwise remove
# it on every pass, leaving each process holding a lock on an unlinked inode --
# which is exactly how two sidecars ended up running at once.
LOCK="$STAGE_ROOT/.${RUN_ID}.sidecar.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another sidecar is already backing up $RUN_ID (lock: $LOCK)." >&2
  echo "Stop it first, or let it keep running -- do not run two." >&2
  exit 3
fi
echo $$ >&9

rsync_args=(-a --delete)
rclone_args=(copy "$STAGE" "$DEST" --update --transfers=4 --checkers=8
             --timeout=300s --retries=3 --low-level-retries=10 --fast-list --stats=0
             --tpslimit "$TPSLIMIT" --drive-pacer-min-sleep 100ms)
[[ -n "$BWLIMIT" ]] && rclone_args+=("--bwlimit=$BWLIMIT")
if [[ -n "$EXCLUDES" ]]; then
  IFS=',' read -ra patterns <<< "$EXCLUDES"
  for pattern in "${patterns[@]}"; do rsync_args+=(--exclude "$pattern"); done
fi

# Verify the destination is usable before claiming to back anything up.
remote_name="${REMOTE%%:*}"
if [[ "$REMOTE" == *:* ]] && ! rclone listremotes 2>/dev/null | grep -qx "$remote_name:"; then
  echo "rclone remote '$remote_name' is not configured." >&2
  echo "Run:  bash tools/connect_drive.sh" >&2
  exit 1
fi

one_pass() {
  rsync "${rsync_args[@]}" "$RUN_DIR/" "$STAGE/" || return 1
  rclone "${rclone_args[@]}"
}

# Ask the remote what it is missing rather than trusting the exit code.
# rclone creates a destination directory before uploading into it, so a pass
# that dies part-way (rate limit, killed process, dropped connection) leaves
# empty datasets/<dataset>/<model>/<template>/ directories behind -- which is
# exactly the "report uploaded, records missing" symptom.
missing_files() {
  rclone check "$STAGE" "$DEST" --one-way --missing-on-dst - \
    --fast-list --stats=0 --tpslimit "$TPSLIMIT" 2>/dev/null
}

verify_and_repair() {
  local attempt missing listing
  listing="$STAGE_ROOT/.${RUN_ID}.missing"
  for attempt in 1 2 3; do
    missing="$(missing_files)"
    if [[ -z "$missing" ]]; then
      echo "[$(date -Is)] verified: remote has every file"
      rm -f "$listing"
      return 0
    fi
    echo "[$(date -Is)] $(wc -l <<< "$missing") file(s) missing on the remote; re-sending (attempt $attempt)" >&2
    printf '%s\n' "$missing" > "$listing"
    rclone copy "$STAGE" "$DEST" --files-from="$listing" --transfers=4 \
      --timeout=300s --retries=5 --low-level-retries=20 --stats=0 \
      --tpslimit "$TPSLIMIT" --drive-pacer-min-sleep 100ms
  done
  missing="$(missing_files)"
  if [[ -n "$missing" ]]; then
    echo "[$(date -Is)] STILL MISSING after 3 repair attempts:" >&2
    printf '%s\n' "$missing" | head -20 >&2
    rm -f "$listing"
    return 1
  fi
  rm -f "$listing"
}

final_pass() {
  echo "[$(date -Is)] final pass"
  one_pass
  verify_and_repair
  echo "[$(date -Is)] stopped"
  exit 0
}
trap final_pass INT TERM

echo "[$(date -Is)] backing up $RUN_DIR -> $DEST every ${INTERVAL}s" \
     "(snapshot via $STAGE; excluding: ${EXCLUDES:-nothing}; ${TPSLIMIT} API calls/s)"
passes=0
failures=0
while true; do
  start=$SECONDS
  if one_pass; then
    passes=$((passes + 1))
    echo "[$(date -Is)] pass $passes ok in $((SECONDS - start))s ($failures failure(s) so far)"
  else
    failures=$((failures + 1))
    echo "[$(date -Is)] pass FAILED (${failures} total) -- retrying in ${INTERVAL}s" >&2
  fi
  sleep "$INTERVAL"
done
