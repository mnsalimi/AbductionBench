#!/usr/bin/env bash
# ONE command to start saving results and logs to the Google Drive folder.
#
#     bash tools/connect_drive.sh
#
# What happens:
#   1. rclone prints a link like  http://127.0.0.1:53682/auth?state=...
#      Ctrl+click it. Because you are attached over VS Code Remote SSH, VS Code
#      forwards that port automatically, so the link opens in your Windows
#      browser. (If it does not: VS Code -> PORTS tab -> Forward a Port -> 53682,
#      then open the link.)
#   2. Approve the Google account that owns the target folder.
#   3. This script writes the rclone remote pinned to that one folder, checks it
#      can read AND write, and then attaches the incremental backup to whatever
#      run is currently in progress.
#
# Nothing needs to be installed on Windows. Re-running this is safe.

set -uo pipefail

FOLDER_ID="${DRIVE_FOLDER_ID:-1BKmNHYIUIBZsnjeDNfpCbGdJYFl5R9i4}"
REMOTE="${DRIVE_REMOTE_NAME:-gdrive}"
CONFIG="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v rclone >/dev/null || { echo "rclone is not installed" >&2; exit 1; }

# Already configured? Skip straight to verification + attaching the backup.
if rclone listremotes 2>/dev/null | grep -qx "$REMOTE:"; then
  echo "remote '$REMOTE' already exists -- verifying it"
else
  cat <<'BANNER'

================================================================================
 A link will appear below. Ctrl+click it (VS Code forwards the port for you),
 then approve the Google account that owns the target Drive folder.
================================================================================

BANNER

  TOKEN_LOG="$(mktemp)"
  trap 'rm -f "$TOKEN_LOG"' EXIT

  # --auth-no-open-browser: there is no browser on this machine, and trying to
  # launch one only prints a confusing error before the link.
  rclone authorize "drive" --auth-no-open-browser 2>&1 | tee "$TOKEN_LOG"

  # rclone prints the token between two markers; take the JSON object itself.
  TOKEN="$(grep -o '{"access_token".*}' "$TOKEN_LOG" | tail -1)"
  if [[ -z "$TOKEN" ]]; then
    echo >&2
    echo "No token was captured -- the authorization did not complete." >&2
    echo "Run this script again, and make sure you open the link and click Allow." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$CONFIG")"
  {
    echo
    echo "[$REMOTE]"
    echo "type = drive"
    echo "scope = drive"
    # Pins the remote to exactly one folder, so nothing can be written anywhere
    # else in the Drive.
    echo "root_folder_id = $FOLDER_ID"
    echo "token = $TOKEN"
  } >> "$CONFIG"
  chmod 600 "$CONFIG"
  echo
  echo "wrote remote '$REMOTE' -> Drive folder $FOLDER_ID"
fi

echo
echo "verifying access..."
if ! rclone lsd "$REMOTE:" >/dev/null 2>&1; then
  echo "  read FAILED -- the account may not have access to folder $FOLDER_ID" >&2
  exit 1
fi
echo "  read  OK"
probe="_abench_write_probe_$$"
if echo ok | rclone rcat "$REMOTE:$probe" 2>/dev/null; then
  echo "  write OK"
  rclone delete "$REMOTE:$probe" 2>/dev/null || true
else
  echo "  write FAILED -- the account needs Editor access to that folder" >&2
  exit 1
fi

# Attach the backup to a run that is already going: a running process cannot
# pick up the remote by itself, and this is the whole point of the exercise.
LATEST_RUN="$(ls -dt "$REPO_DIR"/runs/*/ 2>/dev/null | head -1)"
if [[ -n "$LATEST_RUN" ]] && pgrep -f "[a]bench run" >/dev/null; then
  RUN_ID="$(basename "$LATEST_RUN")"
  if pgrep -f "[s]ync_run.sh" >/dev/null; then
    echo
    echo "a backup sidecar is already running -- leaving it alone"
  else
    echo
    echo "attaching incremental backup to the run in progress ($RUN_ID)"
    nohup bash "$REPO_DIR/tools/sync_run.sh" "$LATEST_RUN" \
      > /tmp/abench_sync.log 2>&1 &
    sleep 6
    tail -3 /tmp/abench_sync.log
    echo
    echo "watch it with:  tail -f /tmp/abench_sync.log"
  fi
else
  echo
  echo "no run in progress; future runs back up on their own"
  echo "(engine.sync is enabled in configs/runs/full.yaml)"
fi

echo
echo "Done. Files land in the Drive folder under one subfolder per run."
