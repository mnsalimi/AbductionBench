#!/usr/bin/env bash
# Stop the 27B: stop tools/qwen_all_parallel.sh (so it cannot restart the
# three-model pass), drain that pass, then start tools/qwen_finish.sh --
# Qwen 2B/4B only, then workbook and final Drive sync. The Qwen servers and
# the Drive sidecar keep running.
#
#   cd /workspace/AbductionBench && bash tools/start_qwen_finish.bash
set -uo pipefail
cd /workspace/AbductionBench || exit 1
LOG=/workspace/abench_trio.log
say() { echo "$(date '+%F %T') [qfin-start] $*" | tee -a "$LOG"; }

for pid in $(pgrep -f 'tools/[q]wen_all_parallel\.sh'); do kill -TERM "$pid" && say "stopped control script $pid"; done
sleep 2
pgrep -f 'tools/[q]wen_all_parallel\.sh' >/dev/null && { say "control script still alive -- not starting"; exit 1; }

run_pid=$(pgrep -f '[b]in/abench run' | head -1)
if [ -n "$run_pid" ]; then
    kill -INT "$run_pid"; say "SIGINT to abench run $run_pid (draining)"
    for i in $(seq 1 36); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    kill -0 "$run_pid" 2>/dev/null && { kill -INT "$run_pid"; say "second SIGINT: cancelling what is still in flight"; }
    for i in $(seq 1 120); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    kill -0 "$run_pid" 2>/dev/null && { say "abench run $run_pid still alive after 13 minutes -- not starting"; exit 1; }
    say "abench run $run_pid exited"
fi
nohup bash tools/qwen_finish.sh > /dev/null 2>&1 &
say "started tools/qwen_finish.sh (pid $!) -- watch: tail -f $LOG | grep qfin"
