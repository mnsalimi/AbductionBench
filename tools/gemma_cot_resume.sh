#!/usr/bin/env bash
# The rest of gemma-4-E4B's and gemma-4-E2B's COT pass (native reasoning ON)
# in 20260924-002252_openrouter-trio, with the GPU to themselves.
#
#   1. waits for any other abench run to end, then stops every vLLM server
#   2. serves gemma-4-E4B (:18000) and gemma-4-E2B (:18005) at maximum
#      throughput: 0.48 + 0.40 of the card, 512 sequences each, 16,384-token
#      scheduler steps, 65,536-token windows, vLLM's gemma4 reasoning parser
#      (native reasoning in its own field). No fallback to anything smaller.
#   3. proves the native reasoning switch on both before generating
#   4. resumes the cot pass for these two (trio_gemma_cot_resume.yaml): every
#      saved answer is reused, only the missing ones are asked; no judge; no
#      built-in Drive sync -- tools/sync_run.sh does that from its own process
#   5. stops both servers and restores their .env files
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/gemma_cot_resume.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log      (steps tagged [gemma-cot])
# Stop:   pkill -f gemma_cot_resume.sh; pkill -INT -f 'bin/abench run'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
CONFIG=configs/runs/trio_gemma_cot_resume.yaml
BACKUP_SUFFIX=.env.gemma-cot-backup

say() { echo "$(date '+%F %T') [gemma-cot] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN"; exit 1; }
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . "$SERVING/qwen3.5-2b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/qwen3.5-2b/.env"; exit 1; }

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
    # A file that does not end in a newline would glue the new line onto its
    # last one (qwen3.5-2b's .env did: ENABLE_PREFIX_CACHING=1 had none).
    [ -n "$(tail -c1 "$file")" ] && echo >> "$file"
    if grep -q "^$2=" "$file"; then sed -i "s|^$2=.*|$2=$3|" "$file"; else echo "$2=$3" >> "$file"; fi
}

healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }

start_service() {  # start_service <service> <port>
    local name=$1 port=$2 waited=0
    supervisorctl start "$name" >>"$LOG" 2>&1
    while [ $waited -lt 1800 ]; do
        if healthy "$port"; then say "$name is up on :$port after ${waited}s"; return 0; fi
        if supervisorctl status "$name" | grep -qE "FATAL|EXITED|BACKOFF"; then
            say "$name failed to start: $(supervisorctl status "$name")"
            tail -5 "/var/log/portal/$name.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
            supervisorctl stop "$name" >>"$LOG" 2>&1
            return 1
        fi
        sleep 15; waited=$((waited + 15))
    done
    say "$name did not answer within 30 minutes"; supervisorctl stop "$name" >>"$LOG" 2>&1; return 1
}

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

native_reasoning() {  # native_reasoning <port> <model> <true|false>  -> "on" / "off" / "no reply"
    local reply
    reply=$(curl -s -m 600 "http://127.0.0.1:$1/v1/chat/completions" \
        -H "Authorization: Bearer $ABENCH_API_KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"A glass of water left in a warm room is empty after three days, and nobody drank from it. Which is more likely: 1) evaporation 2) a leak? Explain briefly, then give the label.\"}],\"max_tokens\":3000,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":$3}}")
    python3 - "$reply" <<'PYEOF'
import json, sys
try:
    msg = json.loads(sys.argv[1])["choices"][0]["message"]
except Exception:
    print("no reply"); sys.exit(0)
native = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
content = msg.get("content") or ""
inline = "<think>" in content or "</think>" in content or "Thinking Process" in content or "<|channel>" in content
print("on" if native else ("inline" if inline else "off"))
PYEOF
}



# -- 0. never two runs in one folder ------------------------------------------
while pgrep -f 'bin/abench run' >/dev/null; do
    say "another abench run is still going; waiting"; sleep 60
done

# -- 1. free the GPU -----------------------------------------------------------
running=$(supervisorctl status | awk '/-vllm/ && /RUNNING/ {print $1}')
for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
say "stopped: ${running:-nothing was running}"
for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
say "GPU memory in use: $(gpu_used) MiB"

# -- 2. serve the two gemmas ---------------------------------------------------
# 0.48 + 0.40 = 0.88 of 95.6 GiB, ~11 GiB left for cuBLAS workspace outside
# vLLM's reservation. The KV pool, not the 512-sequence cap, decides how many
# decode at once.
for m in gemma-4-e4b gemma-4-e2b; do
    set_env $m MAX_MODEL_LEN 65536
    set_env $m MAX_NUM_SEQS 512
    set_env $m MAX_NUM_BATCHED_TOKENS 16384
    set_env $m ENFORCE_EAGER 0
    set_env $m REASONING_PARSER gemma4
done
set_env gemma-4-e4b GPU_MEMORY_UTILIZATION 0.48
set_env gemma-4-e2b GPU_MEMORY_UTILIZATION 0.40
SERVERS="gemma-4-e4b-vllm:18000:google/gemma-4-E4B-it gemma-4-e2b-vllm:18005:google/gemma-4-E2B-it"
stop_all() { for s in $SERVERS; do supervisorctl stop "${s%%:*}" >>"$LOG" 2>&1; done; }
for s in $SERVERS; do
    name=${s%%:*}; rest=${s#*:}; port=${rest%%:*}
    start_service "$name" "$port" || { say "$name will not start with these settings -- stopping, nothing generated"; stop_all; exit 1; }
done

# -- 3. prove the native reasoning switch -------------------------------------
for s in $SERVERS; do
    rest=${s#*:}; port=${rest%%:*}; model=${rest#*:}
    off=$(native_reasoning "$port" "$model" false); on=$(native_reasoning "$port" "$model" true)
    if [ "$off" = "off" ] && [ "$on" = "on" ]; then
        say "probe $model: enable_thinking=false -> no reasoning, true -> native reasoning in its own field -- ok"
    else
        say "probe $model: FAILED -- false gave '$off', true gave '$on' (expected off / on). NOT GENERATING."
        stop_all; exit 2
    fi
done

# -- 4. resume the cot pass ----------------------------------------------------
say "cot pass: abench run $CONFIG --resume $RUN"
.venv/bin/abench run "$CONFIG" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "cot pass exited with $rc"

# -- 5. free the GPU again -----------------------------------------------------
stop_all
say "stopped both gemma servers; nothing is running on the GPU"
exit $rc
