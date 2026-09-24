#!/usr/bin/env bash
# Stop the current Qwen cot run safely, then start tools/qwen_cot_then_27b_openrouter.sh
# detached (phase 1: the same Qwen 2B/4B cot pass; phase 2: Qwen3.5-27B via OpenRouter, io +
# cot; end: workbook rebuilt, final Drive sync).
#
#   cd /workspace/AbductionBench && bash tools/start_qwen_cot_then_27b.bash
#
# Order matters: the wrapper goes first -- if the engine exited first, the
# wrapper would stop the servers and the Drive sidecar on its own. The engine
# gets SIGINT (drain: every saved answer stays saved); if it is still waiting
# on in-flight answers after 3 minutes, a second SIGINT cancels them (those
# are simply asked again by phase 1). The Drive sidecar is left running.
set -uo pipefail
cd /workspace/AbductionBench || exit 1
LOG=/workspace/abench_trio.log
say() { echo "$(date '+%F %T') [q27or-start] $*" | tee -a "$LOG"; }

for pid in $(pgrep -f 'tools/[q]wen_cot_(continue|resume)\.sh'); do
    kill -TERM "$pid" && say "stopped wrapper $pid"
done
sleep 2

run_pid=$(pgrep -f '[b]in/abench run' | head -1)
if [ -n "$run_pid" ]; then
    kill -INT "$run_pid"; say "SIGINT to abench run $run_pid (draining)"
    for i in $(seq 1 36); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$run_pid" 2>/dev/null; then
        kill -INT "$run_pid"; say "second SIGINT: cancelling in-flight answers"
    fi
    for i in $(seq 1 120); do kill -0 "$run_pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$run_pid" 2>/dev/null; then
        say "abench run $run_pid still alive after 13 minutes -- not starting; check it"; exit 1
    fi
    say "abench run $run_pid exited"
else
    say "no abench run was going"
fi

nohup bash tools/qwen_cot_then_27b_openrouter.sh > /dev/null 2>&1 &
say "started tools/qwen_cot_then_27b_openrouter.sh (pid $!) -- watch: tail -f $LOG | grep -E 'q27or|finished'"
