#!/usr/bin/env bash
# Finish all judging in 20260924-002252_openrouter-trio with two judges at
# once -- local gpt-oss-120b (vLLM, :18004) and gpt-oss-120b on CoreWeave via
# OpenRouter -- pulling from one queue (tools/judge_shared_queue.py). LOCAL ONLY:
# nothing is sent to Google Drive.
#
#   0. refuses to start while the old judging (judge_all_after_generation.sh,
#      judge_answers_offline.py, abench judge-reasoning / run) is alive -- two
#      writers on the same records would lose results
#   1. stops the Drive sidecar WITHOUT its final upload (SIGKILL, so it cannot
#      start one)
#   2. makes sure the local judge is up (starts it with the judging settings
#      if not) and answers
#   3. PROBE: 20 real records on each judge; any failure stops here
#   2b. queues again every sample whose judging failed (tools/reset_failed_reasoning.py)
#   4. everything still owed: trio remainder (local), the answer judge for the
#      five new models (local), their reasoning judge (both judges, shared
#      queue; remote at 48 in flight, paused automatically on a failure burst).
#      Resumable: records already judged are skipped on a re-run.
#   4b. one retry pass for what failed during step 4
#   5. judge server stopped, workbook rebuilt locally (abench report)
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/judge_shared_queue.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log | grep -E 'shared-q|judge_shared_queue: (progress|DONE|to judge|estimate|PROBE)'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
JUDGE_SERVICE=gpt-oss-120b-vllm
JUDGE_PORT=18004
BACKUP_SUFFIX=.env.shared-q-backup

say() { echo "$(date '+%F %T') [shared-q] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN"; exit 1; }
set -a; . /workspace/.env; set +a
ABENCH_API_KEY=$(set -a; . "$SERVING/gpt-oss-120b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/gpt-oss-120b/.env"; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { say "no OPENROUTER_API_KEY in /workspace/.env"; exit 1; }

restore_envs() {
    for backup in "$SERVING"/*/"$BACKUP_SUFFIX"; do
        [ -f "$backup" ] || continue
        mv -f "$backup" "$(dirname "$backup")/.env"
        say "restored $(dirname "$backup")/.env"
    done
}
trap restore_envs EXIT

set_env() {  # set_env <model dir> KEY VALUE
    local file="$SERVING/$1/.env"
    [ -f "$SERVING/$1/$BACKUP_SUFFIX" ] || cp -p "$file" "$SERVING/$1/$BACKUP_SUFFIX"
    [ -n "$(tail -c1 "$file")" ] && echo >> "$file"
    if grep -q "^$2=" "$file"; then sed -i "s|^$2=.*|$2=$3|" "$file"; else echo "$2=$3" >> "$file"; fi
}
healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }

# -- 0. no other writer ---------------------------------------------------------
others=$(pgrep -af 'tools/[j]udge_all_after_generation\.sh|tools/[j]udge_answers_offline\.py|[b]in/abench (judge-reasoning|run)|tools/[q]wen_(finish|all_parallel)\.sh' || true)
if [ -n "$others" ]; then
    say "NOT STARTING -- still running (stop these first):"
    echo "$others" | sed 's/^/    /' | tee -a "$LOG"
    exit 1
fi

# -- 1. no Drive -------------------------------------------------------------------
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    pkill -KILL -f "tools/sync_run.sh .*$RUN"
    pkill -KILL -f "rclone copy /tmp/abench_sync_stage/$RUN" 2>/dev/null
    say "Drive sidecar stopped without a final upload (this run stays local)"
fi

# -- 2. the local judge ------------------------------------------------------------
if ! healthy $JUDGE_PORT; then
    set_env gpt-oss-120b GPU_MEMORY_UTILIZATION 0.90
    set_env gpt-oss-120b MAX_MODEL_LEN 65536
    set_env gpt-oss-120b MAX_NUM_SEQS 256
    set_env gpt-oss-120b MAX_NUM_BATCHED_TOKENS 16384
    set_env gpt-oss-120b ENFORCE_EAGER 0
    running=$(supervisorctl status | awk '/-vllm/ && /RUNNING|STARTING/ {print $1}')
    for name in $running; do [ "$name" = "$JUDGE_SERVICE" ] || supervisorctl stop "$name" >>"$LOG" 2>&1; done
    supervisorctl start "$JUDGE_SERVICE" >>"$LOG" 2>&1
    for _ in $(seq 1 160); do healthy $JUDGE_PORT && break; sleep 15; done
fi
healthy $JUDGE_PORT || { say "local judge not answering on :$JUDGE_PORT -- stopping"; exit 1; }
say "local judge up on :$JUDGE_PORT"

# -- 2b. queue again every sample whose judging failed somewhere (timeouts,
#        exhausted retries, unusable step lists); cached families stay free -------
.venv/bin/python tools/reset_failed_reasoning.py "runs/$RUN" --apply >>"$LOG" 2>&1 \
    && say "failed samples queued again (see reset_failed_reasoning output above)" \
    || { say "reset_failed_reasoning skipped files written in the last 2 min -- is another run live? stopping"; exit 1; }

# -- 3. probe both judges ------------------------------------------------------------
# JUDGE_REMOTE=off -> local judge only (no OpenRouter at all): CoreWeave hung
# on long requests from ~14:20 on 2026-09-25.
REMOTE_ARGS=""
[ "${JUDGE_REMOTE:-on}" = "off" ] && REMOTE_ARGS="--no-remote" && say "LOCAL JUDGE ONLY (JUDGE_REMOTE=off): nothing is sent to OpenRouter"
say "probe: 5 real records on the local judge and on the first remote host (${JUDGE_REMOTES:-gpt-oss-120b-coreweave})"
.venv/bin/python tools/judge_shared_queue.py "runs/$RUN" --probe --probe-size 5 \
    --remotes "${JUDGE_REMOTES:-gpt-oss-120b-coreweave}" $REMOTE_ARGS >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "PROBE FAILED (exit $rc) -- nothing more is sent; see the log above"
    exit 1
fi
say "probe passed"

# -- 4. everything still owed ------------------------------------------------------
say "full pass: trio remainder + answer judge (local), reasoning judge (local + CoreWeave, shared queue)"
# Remote at 48 in flight (not 128): CoreWeave timed out in bursts at 128. The
# tool pauses the remote judge for 10 min whenever >= 30 of its requests fail
# permanently in one minute; the local judge carries on.
# JUDGE_REMOTES: which OpenRouter hosts (comma-separated judge ids from
# configs/runs/judge_shared_queue.yaml) share the queue with the local judge.
QUEUE_ARGS="--remote-calls 48 --remote-jobs 12 --remotes ${JUDGE_REMOTES:-gpt-oss-120b-coreweave} $REMOTE_ARGS"
.venv/bin/python tools/judge_shared_queue.py "runs/$RUN" $QUEUE_ARGS >>"$LOG" 2>&1
rc=$?
say "full pass exited with $rc"

# -- 4b. one retry pass for whatever failed during the full pass ------------------
.venv/bin/python tools/reset_failed_reasoning.py "runs/$RUN" --apply >>"$LOG" 2>&1
say "retry pass: samples whose judging failed during the full pass"
.venv/bin/python tools/judge_shared_queue.py "runs/$RUN" $QUEUE_ARGS --no-answers >>"$LOG" 2>&1
rc=$?
say "retry pass exited with $rc$([ $rc -ne 0 ] && echo ' -- some jobs failed; run this script again to retry only what is missing')"

# -- 5. the end (local only) -------------------------------------------------------
supervisorctl stop "$JUDGE_SERVICE" >>"$LOG" 2>&1
say "judge server stopped; the GPU is free"
restore_envs
.venv/bin/abench report "runs/$RUN" >>"$LOG" 2>&1 && say "workbook rebuilt (local): runs/$RUN/reports/" || say "abench report failed (see log)"
say "done -- nothing was sent to Drive"
