#!/usr/bin/env bash
# Remove every result and log of gemma-4-E4B, gemma-4-E2B, Qwen3.5-2B and
# Qwen3.5-4B from 20260924-002252_openrouter-trio -- locally and on Drive.
# The trio (gpt-5.6-luna, gemini-3.8-flash, gemma-4-31b) is not touched.
#
#   bash tools/cleanup_local4.sh           # DRY RUN: prints what it would do, changes nothing
#   bash tools/cleanup_local4.sh --apply   # does it
#
# Local
#   * the set-aside answers: runs/_superseded/<run>__local4_before_native_reasoning
#   * the (now empty) model folders under the run's datasets/
#   * the resolved-config snapshots of their passes (those listing no other model)
#   * their lines in engine.log, engine.jsonl, events.jsonl and
#     /workspace/abench_trio.log -- each original kept once as <file>.before-cleanup
#   * the four vLLM servers' own logs in /var/log/portal
#   * the Drive-sync staging copy (/tmp/abench_sync_stage/<run>), rebuilt next sync
#   * then the reports are rebuilt from what is left (abench report)
# Drive (gdrive:AbductionBench/<run>), deleted files go to Drive's trash
#   * any file left under the four models' folders, then the empty folders
#   * the same resolved-config snapshots
#   * the cleaned logs and rebuilt reports are uploaded over the old ones
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
RUN_DIR=$REPO/runs/$RUN
REMOTE=gdrive:AbductionBench/$RUN
SETASIDE=$REPO/runs/_superseded/${RUN}__local4_before_native_reasoning
MODELS="gemma-4-e4b-local gemma-4-e2b-local qwen3-5-2b-local qwen3-5-4b-local"
# Every way these four appear in a log line: run ids, served names, vLLM
# service names, and the tags of the scripts that ran them.
LINES='gemma-4-e4b|gemma-4-e2b|qwen3[.-]5-2b|qwen3[.-]5-4b|\[gemma-gen\]|\[qwen-gen\]|\[local4\]'
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

do_() { if [ $APPLY -eq 1 ]; then "$@"; else echo "   would run: $*"; fi; }
say() { echo "$(date '+%T') $*"; }

cd "$REPO" || exit 1
if pgrep -f 'bin/abench run' >/dev/null || pgrep -f 'local4_generate.sh' >/dev/null; then
    say "an abench run is going -- stop it first"; exit 1
fi
[ $APPLY -eq 1 ] && say "APPLYING" || say "DRY RUN -- nothing will change (add --apply to do it)"

# ---------------------------------------------------------------- local ----
say "1. set-aside answers: $SETASIDE ($(du -sh "$SETASIDE" 2>/dev/null | cut -f1))"
[ -d "$SETASIDE" ] && do_ rm -rf "$SETASIDE"

dirs=()
for m in $MODELS; do for d in "$RUN_DIR"/datasets/*/"$m"; do [ -d "$d" ] && dirs+=("$d"); done; done
say "2. model folders in the run: ${#dirs[@]} ($(find "${dirs[@]}" -type f 2>/dev/null | wc -l) file(s) inside)"
[ ${#dirs[@]} -gt 0 ] && do_ rm -rf "${dirs[@]}"

snapshots=()
for f in "$RUN_DIR"/run_config.resolved*.yaml; do
    [ "$f" = "$RUN_DIR/run_config.resolved.yaml" ] && continue       # the run's own, never
    # Only a snapshot whose evaluated models are ALL among the four.
    if python3 - "$f" "$MODELS" <<'EOF'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
ids = [m["id"] for m in c.get("models", []) if not m.get("judge_only")]
sys.exit(0 if ids and set(ids) <= set(sys.argv[2].split()) else 1)
EOF
    then snapshots+=("$f"); fi
done
say "3. resolved-config snapshots of their passes: ${snapshots[*]:-none}"
[ ${#snapshots[@]} -gt 0 ] && do_ rm -f "${snapshots[@]}"

say "4. their lines in the shared logs"
for f in "$RUN_DIR/engine.log" "$RUN_DIR/engine.jsonl" "$RUN_DIR/events.jsonl" /workspace/abench_trio.log; do
    [ -f "$f" ] || continue
    n=$(grep -ciE "$LINES" "$f")
    echo "   $f: $n of $(wc -l < "$f") line(s)"
    [ "$n" -gt 0 ] || continue
    if [ $APPLY -eq 1 ]; then
        [ -e "$f.before-cleanup" ] || cp -p "$f" "$f.before-cleanup"
        grep -viE "$LINES" "$f.before-cleanup" > "$f.tmp" && mv "$f.tmp" "$f"
    else
        echo "   would keep the original as $f.before-cleanup and rewrite $f without them"
    fi
done

say "5. the four vLLM servers' logs"
for l in /var/log/portal/gemma-4-e4b-vllm.log /var/log/portal/gemma-4-e2b-vllm.log /var/log/portal/qwen3.5-2b-vllm.log /var/log/portal/qwen3.5-4b-vllm.log; do
    [ -f "$l" ] && { echo "   $l ($(du -h "$l" | cut -f1))"; do_ truncate -s 0 "$l"; }
done

say "6. Drive-sync staging copy"
[ -d "/tmp/abench_sync_stage/$RUN" ] && do_ rm -rf "/tmp/abench_sync_stage/$RUN"

say "7. rebuild the reports from what is left"
do_ .venv/bin/abench report "runs/$RUN"

# ---------------------------------------------------------------- Drive ----
say "8. Drive: anything left under the four models' folders, then the empty folders"
includes=(); for m in $MODELS; do includes+=(--include "/*/$m/**"); done
left=$(timeout 300 rclone lsf -R --files-only "$REMOTE/datasets" "${includes[@]}" 2>/dev/null | wc -l)
echo "   files still there: $left"
[ "$left" -gt 0 ] && do_ rclone delete "$REMOTE/datasets" --drive-use-trash --tpslimit 4 -v "${includes[@]}"
do_ rclone rmdirs "$REMOTE/datasets" --leave-root --tpslimit 4

say "9. Drive: the same resolved-config snapshots"
for f in "${snapshots[@]}"; do do_ rclone deletefile "$REMOTE/$(basename "$f")" --drive-use-trash; done

say "10. Drive: upload the cleaned logs and the rebuilt reports over the old ones"
do_ rclone copy "$RUN_DIR" "$REMOTE" --tpslimit 4 -v \
    --include "/engine.log" --include "/engine.jsonl" --include "/events.jsonl" \
    --include "/RUN_REPORT.md" --include "/reports/**"

say "done$([ $APPLY -eq 1 ] || echo ' (dry run)')"
