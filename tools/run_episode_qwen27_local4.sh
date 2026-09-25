#!/usr/bin/env bash
# qwen3.5-27b (OpenRouter, cheapest host) and gemma-4-E4B/E2B + qwen3.5-4B/2B
# (served here by vLLM, all four at once) on 50 records of every interactive and
# sequential dataset -- ONE run, all five models in parallel
# (configs/runs/episode_qwen27_local4.yaml), then the results workbook.
#
#   1. refuses to start while another run or judging is live
#   2. stops any vLLM service, sets the four small models' servers to 65,536
#      context / 512 sequences / 0.26+0.19+0.24+0.17 of the GPU (the settings
#      tools/local4_generate.sh ran all four with; the .env files are backed up
#      and restored at the end) and starts them
#   3. PROBE: each must answer with NO native reasoning when enable_thinking is
#      false -- if one does not, nothing is run
#   4. the run, with a BUDGET GUARD: OpenRouter's spend for the key is read
#      every 60 s, and the run is stopped if it passes BUDGET_USD (default 8)
#   5. the four servers stopped (GPU free), .env files restored, workbook built
#
# Start:  cd /workspace/AbductionBench && BUDGET_USD=8 nohup bash tools/run_episode_qwen27_local4.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/episode_qwen27_local4.log | grep -E '\[q27l4\]|finished in'
set -uo pipefail

REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
PY=$REPO/.venv/bin/python
LOG=/workspace/episode_qwen27_local4.log
CONFIG=configs/runs/episode_qwen27_local4.yaml
BUDGET_USD=${BUDGET_USD:-8}
DATASETS=medqdx,med_inquire,ddxplus,vivabench,cloud_opsbench,athena_bench
BACKUP_SUFFIX=.env.q27l4-backup
SERVERS="gemma-4-e4b-vllm:18000:google/gemma-4-E4B-it gemma-4-e2b-vllm:18005:google/gemma-4-E2B-it qwen3.5-4b-vllm:18002:Qwen/Qwen3.5-4B qwen3.5-2b-vllm:18001:Qwen/Qwen3.5-2B"

say() { echo "$(date '+%F %T') [q27l4] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
set -a; . /workspace/.env; set +a
ABENCH_API_KEY=$(set -a; . "$SERVING/qwen3.5-2b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY PYTHONPATH=src
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/qwen3.5-2b/.env"; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { say "no OPENROUTER_API_KEY in /workspace/.env"; exit 1; }

# -- 1. nothing else live -------------------------------------------------------
others=$(pgrep -af '[a]bductionbench.cli run|[b]in/abench (run|judge)|tools/[j]udge_shared_queue|tools/[r]un_episode_trio50' || true)
if [ -n "$others" ]; then
    say "NOT STARTING -- another run is live:"; echo "$others" | tee -a "$LOG"; exit 1
fi

restore_envs() {
    for backup in "$SERVING"/*/"$BACKUP_SUFFIX"; do
        [ -f "$backup" ] || continue
        mv -f "$backup" "$(dirname "$backup")/.env"
        say "restored $(dirname "$backup")/.env"
    done
}
stop_all() { for s in $SERVERS; do supervisorctl stop "${s%%:*}" >>"$LOG" 2>&1; done; }
cleanup() { stop_all; restore_envs; }
trap cleanup EXIT

set_env() {  # set_env <model dir> KEY VALUE
    local file="$SERVING/$1/.env"
    [ -f "$SERVING/$1/$BACKUP_SUFFIX" ] || cp -p "$file" "$SERVING/$1/$BACKUP_SUFFIX"
    [ -n "$(tail -c1 "$file")" ] && echo >> "$file"
    if grep -q "^$2=" "$file"; then sed -i "s|^$2=.*|$2=$3|" "$file"; else echo "$2=$3" >> "$file"; fi
}
healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
start_service() {  # start_service <service> <port>
    local name=$1 port=$2 waited=0
    supervisorctl start "$name" >>"$LOG" 2>&1
    while [ $waited -lt 1800 ]; do
        if healthy "$port"; then say "$name is up on :$port after ${waited}s"; return 0; fi
        if supervisorctl status "$name" | grep -qE "FATAL|EXITED|BACKOFF"; then
            say "$name failed to start: $(supervisorctl status "$name")"
            tail -5 "/var/log/portal/$name.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
            return 1
        fi
        sleep 15; waited=$((waited + 15))
    done
    say "$name did not answer within 30 minutes"; return 1
}
reasoning_off() {  # reasoning_off <port> <model> -> "off" / "on" / "no reply"
    local reply
    reply=$(curl -s -m 600 "http://127.0.0.1:$1/v1/chat/completions" \
        -H "Authorization: Bearer $ABENCH_API_KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"A glass of water left in a warm room is empty after three days, and nobody drank from it. Which is more likely: 1) evaporation 2) a leak? Give the label.\"}],\"max_tokens\":1000,\"temperature\":0.7,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    python3 - "$reply" <<'PYEOF'
import json, sys
try:
    msg = json.loads(sys.argv[1])["choices"][0]["message"]
except Exception:
    print("no reply"); sys.exit(0)
native = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
content = msg.get("content") or ""
inline = "<think>" in content or "</think>" in content or "Thinking Process" in content or "<|channel>" in content
print("on" if (native or inline) else "off")
PYEOF
}
spend() {
    "$PY" - <<'EOF'
import json, os, urllib.request
req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                             headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
print(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])
EOF
}

# -- 2. the four local servers ---------------------------------------------------
running=$(supervisorctl status | awk '/-vllm/ && /RUNNING|STARTING/ {print $1}')
for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
say "stopped: ${running:-nothing was running}"
for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
say "GPU memory in use: $(gpu_used) MiB"
for m in gemma-4-e4b gemma-4-e2b qwen3.5-2b qwen3.5-4b; do
    set_env $m MAX_MODEL_LEN 65536
    set_env $m MAX_NUM_SEQS 512
    set_env $m MAX_NUM_BATCHED_TOKENS 16384
    set_env $m ENFORCE_EAGER 0
done
set_env gemma-4-e4b GPU_MEMORY_UTILIZATION 0.26; set_env gemma-4-e4b REASONING_PARSER gemma4
set_env gemma-4-e2b GPU_MEMORY_UTILIZATION 0.19; set_env gemma-4-e2b REASONING_PARSER gemma4
set_env qwen3.5-4b  GPU_MEMORY_UTILIZATION 0.24; set_env qwen3.5-4b  REASONING_PARSER qwen3; set_env qwen3.5-4b ENABLE_PREFIX_CACHING 1
set_env qwen3.5-2b  GPU_MEMORY_UTILIZATION 0.17; set_env qwen3.5-2b  REASONING_PARSER qwen3; set_env qwen3.5-2b ENABLE_PREFIX_CACHING 1
for s in $SERVERS; do
    name=${s%%:*}; rest=${s#*:}; port=${rest%%:*}
    start_service "$name" "$port" || { say "$name will not start -- stopping, nothing run"; exit 1; }
done
say "all four servers up; GPU memory in use: $(gpu_used) MiB"

# -- 3. reasoning must be off ------------------------------------------------------
for s in $SERVERS; do
    rest=${s#*:}; port=${rest%%:*}; model=${rest#*:}
    got=$(reasoning_off "$port" "$model")
    if [ "$got" != "off" ]; then
        say "PROBE FAILED: $model with enable_thinking=false gave '$got' -- nothing run"; exit 1
    fi
    say "probe $model: enable_thinking=false -> no reasoning -- ok"
done

# -- 4. the run, under the budget guard ---------------------------------------------
START=$(spend) || { say "cannot read OpenRouter spend -- not starting"; exit 1; }
say "start: key spend \$$START, budget for this run \$$BUDGET_USD"
"$PY" -m abductionbench.cli run "$CONFIG" -d "$DATASETS" >> "$LOG" 2>&1 &
RUN=$!
say "run started (pid $RUN): 35 tasks, 5 models in parallel"
while kill -0 "$RUN" 2>/dev/null; do
    sleep 60
    NOW=$(spend) || continue
    say "spent so far \$$("$PY" -c "print(round($NOW - $START, 3))") of \$$BUDGET_USD"
    if "$PY" -c "import sys; sys.exit(0 if $NOW - $START > $BUDGET_USD else 1)"; then
        say "BUDGET REACHED -- stopping the run"
        kill -INT "$RUN" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$RUN" 2>/dev/null || break; sleep 2; done
        kill -TERM "$RUN" 2>/dev/null
    fi
done
wait "$RUN"; rc=$?
END=$(spend) || END=$START
say "run exited with $rc; this run spent \$$("$PY" -c "print(round($END - $START, 3))")"

# -- 5. GPU free, envs back, workbook -----------------------------------------------
stop_all; restore_envs
say "the four servers are stopped; the GPU is free"
RUN_DIR=$(ls -dt runs/*_episode-qwen27-local4 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
    "$PY" tools/episode_results_excel.py "$RUN_DIR" >> "$LOG" 2>&1 \
        && say "WORKBOOK: $REPO/$RUN_DIR/reports/episode_results.xlsx" \
        || say "workbook failed (see log)"
fi
say "done -- nothing was sent to Drive"
