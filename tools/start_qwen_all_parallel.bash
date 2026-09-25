#!/usr/bin/env bash
# Stop the current Qwen 2B/4B pass and its control script, then start
# tools/qwen_all_parallel.sh detached: the two local Qwens and Qwen3.5-27B via
# OpenRouter generating at the same time.
#
#   cd /workspace/AbductionBench && bash tools/start_qwen_all_parallel.bash
#
# The control script goes first -- left alone, it would stop the servers the
# moment its run exits. The vLLM servers and the Drive sidecar keep running.
# The run gets SIGINT (drain: everything answered stays saved); after 3
# minutes a second SIGINT cancels the answers still being generated -- the
# new pass asks those again.
set -uo pipefail
cd /workspace/AbductionBench || exit 1
LOG=/workspace/abench_trio.log
say() { echo "$(date '+%F %T') [allq-start] $*" | tee -a "$LOG"; }

for pid in $(pgrep -f 'tools/[a]fter_27b_phase1\.sh|tools/[q]wen_cot_then_27b|tools/[q]wen_cot_(continue|resume)\.sh'); do
    kill -TERM "$pid" && say "stopped control script $pid"
done
sleep 2
for pid in $(pgrep -f 'tools/[a]fter_27b_phase1\.sh|tools/[q]wen_cot_then_27b'); do
    say "control script $pid still alive -- not starting; check it"; exit 1
done

run_pid=$(pgrep -f '[b]in/abench run' | head -1)
if [ -n "$run_pid" ]; then
    kill -INT "$run_pid"; say "SIGINT to abench run $run_pid (draining)"
    for i in $(seq 1 36); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$run_pid" 2>/dev/null; then
        kill -INT "$run_pid"; say "second SIGINT: cancelling answers still being generated"
    fi
    for i in $(seq 1 120); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$run_pid" 2>/dev/null; then
        say "abench run $run_pid still alive after 13 minutes -- not starting; check it"; exit 1
    fi
    say "abench run $run_pid exited"
fi

nohup bash tools/qwen_all_parallel.sh > /dev/null 2>&1 &
say "started tools/qwen_all_parallel.sh (pid $!) -- watch: tail -f $LOG | grep allq"
