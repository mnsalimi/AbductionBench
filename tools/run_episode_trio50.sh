#!/usr/bin/env bash
# gemma-4-31b-it, gpt-5.6-luna and gemini-3.8-flash (OpenRouter, cheapest
# hosts) on 50 records of every interactive and sequential dataset
# (configs/runs/episode_trio50.yaml), then the results workbook.
#
# BUDGET GUARD: OpenRouter's own spend counter for the key is read at the start
# and every 60 s; if the run has spent more than BUDGET_USD (default 15) it is
# stopped (SIGINT, then SIGTERM). Nothing already written is lost, and
# `--resume <run id>` continues it later. The counter is for the whole key, so
# anything else using the key at the same time counts too.
#
# Start:  cd /workspace/AbductionBench-interseq && nohup bash tools/run_episode_trio50.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/episode_trio50.log | grep -E 'trio50|finished in'
set -uo pipefail

WT=/workspace/AbductionBench-interseq
PY=/workspace/AbductionBench/.venv/bin/python
LOG=/workspace/episode_trio50.log
BUDGET_USD=${BUDGET_USD:-15}
DATASETS=medqdx,med_inquire,ddxplus,vivabench,cloud_opsbench,athena_bench

say() { echo "$(date '+%F %T') [trio50] $*" | tee -a "$LOG"; }

cd "$WT" || exit 1
set -a; . /workspace/.env; set +a
export ABENCH_API_KEY=unused PYTHONPATH=src
[ -n "${OPENROUTER_API_KEY:-}" ] || { say "no OPENROUTER_API_KEY in /workspace/.env"; exit 1; }

others=$(pgrep -af '[a]bductionbench.cli run|[b]in/abench (run|judge)|tools/[j]udge_shared_queue' || true)
if [ -n "$others" ]; then
    say "NOT STARTING -- another run is live:"; echo "$others" | tee -a "$LOG"; exit 1
fi

spend() {
    "$PY" - <<'EOF'
import json, os, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                             headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
print(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
EOF
}

START=$(spend) || { say "cannot read OpenRouter spend -- not starting"; exit 1; }
say "start: key spend \$$START, budget for this run \$$BUDGET_USD"

"$PY" -m abductionbench.cli run configs/runs/episode_trio50.yaml -d "$DATASETS" >> "$LOG" 2>&1 &
RUN=$!
say "run started (pid $RUN)"

while kill -0 "$RUN" 2>/dev/null; do
    sleep 60
    NOW=$(spend) || continue
    USED=$("$PY" -c "print(round($NOW - $START, 3))")
    say "spent so far \$$USED of \$$BUDGET_USD"
    if "$PY" -c "import sys; sys.exit(0 if $NOW - $START > $BUDGET_USD else 1)"; then
        say "BUDGET REACHED -- stopping the run"
        kill -INT "$RUN" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$RUN" 2>/dev/null || break; sleep 2; done
        kill -TERM "$RUN" 2>/dev/null
    fi
done
wait "$RUN"; rc=$?

RUN_DIR=$(ls -dt runs/*_episode-trio50 2>/dev/null | head -1)
END=$(spend) || END=$START
say "run exited with $rc; this run spent \$$("$PY" -c "print(round($END - $START, 3))")"
if [ -n "$RUN_DIR" ]; then
    "$PY" tools/episode_results_excel.py "$RUN_DIR" >> "$LOG" 2>&1 \
        && say "WORKBOOK: $WT/$RUN_DIR/reports/episode_results.xlsx" \
        || say "workbook failed (see log)"
fi
say "done -- nothing was sent to Drive"
