#!/usr/bin/env bash
# Configure the "gdrive" rclone remote used by engine.sync, pinned to one
# Google Drive folder.
#
# This box has no browser, so the OAuth step has to happen on a machine that
# does. That is rclone's documented headless flow and takes one minute:
#
#   1. On your laptop (with rclone installed: https://rclone.org/install/), run
#
#          rclone authorize "drive" '{"scope":"drive"}'
#
#      A browser opens; approve the Google account that owns the target folder.
#      rclone prints a JSON token between two "---" markers, e.g.
#          {"access_token":"ya29...","token_type":"Bearer","refresh_token":"1//...","expiry":"..."}
#
#   2. Back here, paste it as the single argument to this script (in single
#      quotes so the shell leaves the JSON alone):
#
#          bash tools/setup_drive_remote.sh '{"access_token":"...","refresh_token":"...","expiry":"..."}'
#
# The refresh_token keeps working, so this is a one-time step; rclone renews the
# access token by itself for the life of the machine.
#
# Optional but recommended for heavy use: create your own OAuth client
# (https://rclone.org/drive/#making-your-own-client-id) and pass its id/secret
# as $2/$3 -- rclone's shared default client is rate-limited across all its
# users, which can slow uploads during a long run.
#
# Alternative for a Shared Drive: a service account avoids OAuth entirely. Put
# its JSON key at ~/.config/rclone/gdrive-sa.json, share the Drive folder with
# the service account's email as Editor, and replace the token line below with
#     service_account_file = /root/.config/rclone/gdrive-sa.json
# Note that this does NOT work reliably for a folder in someone's personal
# My Drive: files would be owned by the service account, which has no storage
# quota of its own, and uploads fail with storageQuotaExceeded. Use OAuth above
# for a personal-Drive folder, or a Shared Drive for the service account.

set -euo pipefail

TOKEN="${1:-}"
CLIENT_ID="${2:-}"
CLIENT_SECRET="${3:-}"

# The target folder: https://drive.google.com/drive/folders/1BKmNHYIUIBZsnjeDNfpCbGdJYFl5R9i4
# root_folder_id makes "gdrive:" resolve to exactly that folder, so nothing can
# be written anywhere else in the Drive.
FOLDER_ID="${DRIVE_FOLDER_ID:-1BKmNHYIUIBZsnjeDNfpCbGdJYFl5R9i4}"
REMOTE="${DRIVE_REMOTE_NAME:-gdrive}"
CONFIG="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"

if [[ -z "$TOKEN" ]]; then
  echo "usage: bash tools/setup_drive_remote.sh '<token-json>' [client_id] [client_secret]" >&2
  echo >&2
  echo "Get <token-json> on a machine with a browser:" >&2
  echo "    rclone authorize \"drive\" '{\"scope\":\"drive\"}'" >&2
  exit 2
fi

if ! command -v rclone >/dev/null; then
  echo "rclone is not installed on this machine" >&2
  exit 1
fi

if ! printf '%s' "$TOKEN" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
  echo "the token argument is not valid JSON -- paste the whole {...} block, in single quotes" >&2
  exit 2
fi

mkdir -p "$(dirname "$CONFIG")"
if [[ -f "$CONFIG" ]] && grep -q "^\[$REMOTE\]" "$CONFIG"; then
  cp "$CONFIG" "$CONFIG.bak.$(date +%s)"
  echo "note: an existing [$REMOTE] section was found; the old config was backed up" >&2
  python3 - "$CONFIG" "$REMOTE" <<'PY'
import re, sys
path, remote = sys.argv[1], sys.argv[2]
text = open(path).read()
# drop the existing section so the new one below replaces it
text = re.sub(rf"^\[{re.escape(remote)}\]\n(?:(?!\[).*\n)*", "", text, flags=re.MULTILINE)
open(path, "w").write(text)
PY
fi

{
  echo
  echo "[$REMOTE]"
  echo "type = drive"
  echo "scope = drive"
  echo "root_folder_id = $FOLDER_ID"
  [[ -n "$CLIENT_ID" ]] && echo "client_id = $CLIENT_ID"
  [[ -n "$CLIENT_SECRET" ]] && echo "client_secret = $CLIENT_SECRET"
  echo "token = $TOKEN"
} >> "$CONFIG"
chmod 600 "$CONFIG"

echo "wrote $REMOTE: -> Drive folder $FOLDER_ID  ($CONFIG)"
echo
echo "verifying..."
rclone lsd "$REMOTE:" >/dev/null && echo "  read  OK"
probe="_abench_write_probe_$$"
if echo ok | rclone rcat "$REMOTE:$probe" 2>/dev/null; then
  echo "  write OK"
  rclone delete "$REMOTE:$probe" 2>/dev/null || true
else
  echo "  write FAILED -- the account may lack Editor access to that folder" >&2
  exit 1
fi
echo
echo "Ready. The full run backs up automatically (engine.sync in configs/runs/full.yaml):"
echo "    abench run configs/runs/full.yaml"
echo
echo "For a run already in progress, attach the sidecar instead:"
echo "    bash tools/sync_run.sh runs/<run-id>"
