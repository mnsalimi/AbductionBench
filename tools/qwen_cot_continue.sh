#!/usr/bin/env bash
# Continue the Qwens' cot pass on servers that are ALREADY up (started by
# tools/qwen_cot_resume.sh), with a short request queue
# (trio_qwen_cot_resume_v2.yaml), then free the GPU and let the Drive sidecar
# make its final upload. Saved answers are reused.
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/qwen_cot_continue.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log      (steps tagged [qwen-cot2])
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
LOG=/workspace/abench_trio.log
CONFIG=configs/runs/trio_qwen_cot_resume_v2.yaml
say() { echo "$(date '+%F %T') [qwen-cot2] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . /workspace/vllm_serving/qwen3.5-2b/.env; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY

while pgrep -f 'bin/abench run' >/dev/null; do say "waiting for the previous abench run to exit"; sleep 30; done
for port in 18001 18002; do
    curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$port/v1/models" >/dev/null \
        || { say "no server answering on :$port -- start them with tools/qwen_cot_resume.sh instead"; exit 1; }
done

say "cot pass (short queue): abench run $CONFIG --resume $RUN"
.venv/bin/abench run "$CONFIG" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "cot pass exited with $rc"

supervisorctl stop qwen3.5-2b-vllm qwen3.5-4b-vllm >>"$LOG" 2>&1
say "stopped both qwen servers; nothing is running on the GPU"
if pgrep -f "tools/sync_run.sh" >/dev/null; then
    pkill -TERM -f "tools/sync_run.sh"
    say "sidecar asked to stop: it makes its final verified upload, then exits"
fi
exit $rc
