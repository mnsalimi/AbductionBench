#!/usr/bin/env bash
# Two jobs at once in 20260924-002252_openrouter-trio, LOCAL ONLY (no Drive):
#
#   A. the judging, resumed: tools/judge_shared_queue.sh (local gpt-oss-120b +
#      CoreWeave, shared queue, failed samples re-queued, remote paused on
#      failure bursts, workbook rebuilt at its end)
#   B. TypeSafe jev on the SCS selection tasks (configs/runs/trio_jev_scs.yaml):
#      9 tasks x 150 = 1,350 requests to OpenRouter's /v1/systemone -- the same
#      samples every other model answered, 3 repeats each, one jev column (its
#      prompt mode is n/a), no judges. Its records go to its own
#      datasets/*/jev-openrouter/ folders, which the judging never touches.
#
# A starts first: its safety check refuses to run beside another abench run,
# so B waits until A is past that check. Both append to the run's shared logs,
# never to each other's records.
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/judge_and_jev.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log | grep -E 'shared-q|jev|progress|PAUSED'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
LOG=/workspace/abench_trio.log
JEV_CONFIG=configs/runs/trio_jev_scs.yaml
say() { echo "$(date '+%F %T') [jev] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
set -a; . /workspace/.env; set +a
export OPENROUTER_API_KEY
[ -n "${OPENROUTER_API_KEY:-}" ] || { say "no OPENROUTER_API_KEY in /workspace/.env"; exit 1; }

# -- A: the judging -------------------------------------------------------------------
start_line=$(wc -l < "$LOG")
nohup bash tools/judge_shared_queue.sh > /dev/null 2>&1 &
judge_pid=$!
say "started the judging (tools/judge_shared_queue.sh, pid $judge_pid)"

# Wait until A is past its "no other writer" check -- or has refused.
for _ in $(seq 1 180); do
    new=$(tail -n +"$((start_line + 1))" "$LOG")
    if echo "$new" | grep -q "\[shared-q\] local judge up"; then break; fi
    if echo "$new" | grep -qE "\[shared-q\] (NOT STARTING|local judge not answering|no VLLM_API_KEY|no OPENROUTER)"; then
        say "the judging did not start (see [shared-q] lines above) -- jev not started either"; exit 1
    fi
    kill -0 "$judge_pid" 2>/dev/null || { say "the judging exited early -- jev not started"; exit 1; }
    sleep 5
done

# -- B: jev on the SCS tasks -----------------------------------------------------------
say "jev on the SCS selection tasks: abench run $JEV_CONFIG --resume $RUN"
.venv/bin/abench run "$JEV_CONFIG" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "jev pass exited with $rc ($(ls -d runs/$RUN/datasets/*/jev-openrouter/*/ 2>/dev/null | wc -l) jev task folder(s))"
say "the judging continues on its own (pid $judge_pid); it rebuilds the workbook when it ends"
