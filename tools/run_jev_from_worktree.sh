#!/usr/bin/env bash
# Jev on the SCS selection tasks of 20260924-002252_openrouter-trio, run with
# the code of the jev-na-prompt-mode worktree (/workspace/AbductionBench-jevfix)
# so its rows record prompt mode n/a (one jev column) -- while the judging keeps
# running on main's code untouched. Merge jev-na-prompt-mode into main after
# the judging has finished. LOCAL ONLY: sync is off in the config.
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/run_jev_from_worktree.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log | grep -E '\[jev\]|jev-openrouter'
set -uo pipefail
RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
WT=/workspace/AbductionBench-jevfix
LOG=/workspace/abench_trio.log
say() { echo "$(date '+%F %T') [jev] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -f "$WT/src/abductionbench/core/engine.py" ] || { say "no worktree at $WT"; exit 1; }
grep -q 'model.has_prompt_mode else "n/a"),' "$WT/src/abductionbench/core/engine.py" \
    || { say "the worktree lacks the n/a fix -- not starting"; exit 1; }
set -a; . /workspace/.env; set +a
export OPENROUTER_API_KEY
say "jev on the SCS tasks (worktree code): run $WT/configs/runs/trio_jev_scs.yaml --resume $RUN"
PYTHONPATH="$WT/src" .venv/bin/python -m abductionbench.cli run "$WT/configs/runs/trio_jev_scs.yaml" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
n=$(ls -d runs/$RUN/datasets/*/jev-openrouter/*/ 2>/dev/null | wc -l)
say "jev pass exited with $rc: $n jev task folder(s) (expected 9)"
