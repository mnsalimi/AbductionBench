#!/usr/bin/env bash
# Qwen3.5-2B + Qwen3.5-4B, GENERATION ONLY, into 20260924-002252_openrouter-trio.
#
#   1. waits for any other abench run in this folder to end, then stops every
#      vLLM server on this box
#   2. serves Qwen3.5-2B (:18001) and Qwen3.5-4B (:18002) side by side at
#      maximum throughput: 0.36 + 0.52 of the card, 512 sequences each,
#      16,384-token scheduler steps, 65,536-token windows -- every dataset gets
#      the full 32,000-token answer budget. No fallback to anything smaller.
#   3. checks both answer without thinking when enable_thinking is false
#   4. io pass (thinking off for both), then cot pass (each model's default:
#      the 4B thinks, the 2B does not) -- every dataset, no judge, synced to Drive
#   5. stops both servers and puts their .env files back
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/qwen_small_generate.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log      (steps tagged [qwen-gen])
# Stop:   pkill -f qwen_small_generate.sh; pkill -INT -f 'bin/abench run'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
CONFIG_IO=configs/runs/trio_qwen_small_generate_io.yaml
CONFIG_COT=configs/runs/trio_qwen_small_generate.yaml
BACKUP_SUFFIX=.env.qwen-gen-backup

say() { echo "$(date '+%F %T') [qwen-gen] $*" | tee -a "$LOG"; }

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

thinking_is_off() {  # thinking_is_off <port> <model name>
    local reply
    reply=$(curl -s -m 300 "http://127.0.0.1:$1/v1/chat/completions" \
        -H "Authorization: Bearer $ABENCH_API_KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"A man comes home soaked and the street is wet. Most likely cause: 1) it rained 2) he swam. Answer with only the label.\"}],\"max_tokens\":400,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}")
    python3 - "$reply" <<'EOF'
import json, sys
try:
    msg = json.loads(sys.argv[1])["choices"][0]["message"]
except Exception:
    print("no reply"); sys.exit(1)
content = msg.get("content") or ""
native = msg.get("reasoning") or msg.get("reasoning_content") or ""
print(repr(content[:80]))
sys.exit(1 if ("<think>" in content or "Thinking Process" in content or native.strip()) else 0)
EOF
}

# -- 0. never two runs in one folder ------------------------------------------
while pgrep -f 'bin/abench run' >/dev/null || pgrep -f "rclone copy .*$RUN" >/dev/null; do
    say "another abench run or its upload is still going; waiting"; sleep 60
done

# -- 1. free the GPU -----------------------------------------------------------
running=$(supervisorctl status | awk '/-vllm/ && /RUNNING/ {print $1}')
for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
say "stopped: ${running:-nothing was running}"
for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
say "GPU memory in use: $(gpu_used) MiB"

# -- 2. serve the two qwens ----------------------------------------------------
# The 4B's service was restored on 2026-09-24; supervisor has to be told.
supervisorctl reread >>"$LOG" 2>&1; supervisorctl update qwen3.5-4b-vllm >>"$LOG" 2>&1
# 0.36 + 0.52 = 0.88 of 95.6 GiB, ~11 GiB left for cuBLAS workspace outside
# vLLM's reservation. Both are hybrid linear-attention models, so their KV is
# cheap; 512 sequences each, 16,384-token scheduler steps, prefix caching on
# (every prompt is asked three times, as repeats).
for m in qwen3.5-2b qwen3.5-4b; do
    set_env $m MAX_MODEL_LEN 65536
    set_env $m MAX_NUM_SEQS 512
    set_env $m MAX_NUM_BATCHED_TOKENS 16384
    set_env $m ENABLE_PREFIX_CACHING 1
    set_env $m ENFORCE_EAGER 0
done
set_env qwen3.5-2b GPU_MEMORY_UTILIZATION 0.36
set_env qwen3.5-4b GPU_MEMORY_UTILIZATION 0.52
start_service qwen3.5-4b-vllm 18002 || { say "Qwen3.5-4B will not start with these settings -- stopping, nothing generated"; exit 1; }
start_service qwen3.5-2b-vllm 18001 || { say "Qwen3.5-2B will not start with these settings -- stopping, nothing generated"; supervisorctl stop qwen3.5-4b-vllm >>"$LOG" 2>&1; exit 1; }

# -- 3. io must not think ------------------------------------------------------
for probe in "18001 Qwen/Qwen3.5-2B" "18002 Qwen/Qwen3.5-4B"; do
    # shellcheck disable=SC2086
    if thinking_is_off $probe >>"$LOG" 2>&1; then
        say "probe ${probe#* }: answers without thinking -- ok"
    else
        say "probe ${probe#* }: THINKING IS ON -- stopping before any generation"
        supervisorctl stop qwen3.5-2b-vllm qwen3.5-4b-vllm >>"$LOG" 2>&1
        exit 1
    fi
done

# -- 4. generate: io (thinking off), then cot (each model's default) ------------
say "io pass: abench run $CONFIG_IO --resume $RUN"
.venv/bin/abench run "$CONFIG_IO" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "io pass exited with $rc"
if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then
    while pgrep -f 'bin/abench run' >/dev/null; do sleep 30; done
    say "cot pass: abench run $CONFIG_COT --resume $RUN"
    .venv/bin/abench run "$CONFIG_COT" --resume "$RUN" >>"$LOG" 2>&1
    rc=$?
    say "cot pass exited with $rc"
else
    say "not starting the cot pass"
fi

# -- 5. free the GPU again -----------------------------------------------------
supervisorctl stop qwen3.5-2b-vllm qwen3.5-4b-vllm >>"$LOG" 2>&1
say "stopped both qwen servers; nothing is running on the GPU"
exit $rc
