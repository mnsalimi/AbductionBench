#!/usr/bin/env bash
# Qwen3.5-2B / 4B only: finish their cot pass on the vLLM servers that are
# already up, then stop the servers, rebuild the workbook and make the final
# verified Drive sync. No 27B, no judges.
#
# Start:  bash tools/start_qwen_finish.bash     (stops the three-model pass first)
# Watch:  tail -f /workspace/abench_trio.log | grep qfin
set -uo pipefail
RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
CONFIG=configs/runs/trio_qwen_cot_finish.yaml
SIDECAR="$REPO/tools/sync_run.sh"
say() { echo "$(date '+%F %T') [qfin] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . "$SERVING/qwen3.5-2b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }

while pgrep -f '[b]in/abench run' >/dev/null || pgrep -f 'tools/[q]wen_all_parallel\.sh' >/dev/null; do
    say "another run or control script is still going; waiting"; sleep 30
done
for port in 18001 18002; do
    healthy $port || { say "no Qwen server answering on :$port -- not starting (start them with tools/qwen_all_parallel.sh's settings)"; exit 1; }
done
if ! pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    say "started the Drive sync sidecar"
fi

say "Qwen 2B/4B cot pass (no 27B): abench run $CONFIG --resume $RUN"
.venv/bin/abench run "$CONFIG" --resume "$RUN" >>"$LOG" 2>&1
say "Qwen cot pass exited with $?"

supervisorctl stop qwen3.5-2b-vllm qwen3.5-4b-vllm >>"$LOG" 2>&1
say "stopped both Qwen servers; the GPU is free"
say "rebuilding the workbook from every record on disk"
.venv/bin/abench report "runs/$RUN" >>"$LOG" 2>&1 && say "workbook rebuilt" || say "abench report failed (see log)"
say "Drive: final sync"
pkill -TERM -f "tools/sync_run.sh .*$RUN"
for _ in $(seq 1 720); do pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null || break; sleep 10; done
pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null && say "Drive sync still going after 2 hours (log /tmp/abench_sync.log)" || say "Drive sync finished. All done."
