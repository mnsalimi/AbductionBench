#!/usr/bin/env bash
# gemma-4-E4B + gemma-4-E2B, GENERATION ONLY, into 20260924-002252_openrouter-trio.
#
#   1. stops every vLLM server on this box (the gpt-oss-120b judge included)
#   2. serves gemma-4-E4B (:18000) and gemma-4-E2B (:18005) side by side:
#      0.45 + 0.35 of the card, 256 sequences each, 32,768-token windows
#      (falling back to 16,384 if a server will not start that way)
#   3. checks each answers io without thinking
#   4. runs configs/runs/trio_gemma_small_generate.yaml --resume: io and cot on
#      every dataset, no judge, synced to Drive
#   5. stops both servers and puts their .env files back
#
# Nothing is restarted afterwards: bring the judge back when you choose.
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/gemma_small_generate.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log      (steps tagged [gemma-gen])
# Stop:   pkill -f gemma_small_generate.sh; pkill -INT -f 'bin/abench run'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
CONFIG=configs/runs/trio_gemma_small_generate.yaml
BACKUP_SUFFIX=.env.gemma-gen-backup

say() { echo "$(date '+%F %T') [gemma-gen] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN"; exit 1; }
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . "$SERVING/gemma-4-e4b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/gemma-4-e4b/.env"; exit 1; }

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

# -- 2. serve the two gemmas ---------------------------------------------------
# 0.45 + 0.35 = 0.80 of 95.6 GiB, ~19 GiB left for cuBLAS workspace outside
# vLLM's reservation (0.78 beside another model once died in service for want
# of it). 256 sequences each; the KV pool decides how many actually run.
set_env gemma-4-e4b GPU_MEMORY_UTILIZATION 0.45; set_env gemma-4-e4b MAX_MODEL_LEN 32768; set_env gemma-4-e4b MAX_NUM_SEQS 256
set_env gemma-4-e2b GPU_MEMORY_UTILIZATION 0.35; set_env gemma-4-e2b MAX_MODEL_LEN 32768; set_env gemma-4-e2b MAX_NUM_SEQS 256
export ABENCH_E4B_WINDOW=32768 ABENCH_E2B_WINDOW=32768
if ! start_service gemma-4-e4b-vllm 18000; then
    say "retrying gemma-4-E4B with its old 16,384 window"
    set_env gemma-4-e4b MAX_MODEL_LEN 16384; export ABENCH_E4B_WINDOW=16384
    start_service gemma-4-e4b-vllm 18000 || { say "gemma-4-E4B will not start -- stopping"; exit 1; }
fi
if ! start_service gemma-4-e2b-vllm 18005; then
    say "retrying gemma-4-E2B with its old 16,384 window"
    set_env gemma-4-e2b MAX_MODEL_LEN 16384; export ABENCH_E2B_WINDOW=16384
    start_service gemma-4-e2b-vllm 18005 || { say "gemma-4-E2B will not start -- stopping"; supervisorctl stop gemma-4-e4b-vllm; exit 1; }
fi

# -- 3. io must not think ------------------------------------------------------
for probe in "18000 google/gemma-4-E4B-it" "18005 google/gemma-4-E2B-it"; do
    # shellcheck disable=SC2086
    if thinking_is_off $probe >>"$LOG" 2>&1; then
        say "probe ${probe#* }: answers without thinking -- ok"
    else
        say "probe ${probe#* }: THINKING IS ON -- stopping before any generation"
        supervisorctl stop gemma-4-e4b-vllm gemma-4-e2b-vllm >>"$LOG" 2>&1
        exit 1
    fi
done

# -- 4. generate ---------------------------------------------------------------
say "generating: abench run $CONFIG --resume $RUN (windows: E4B $ABENCH_E4B_WINDOW, E2B $ABENCH_E2B_WINDOW)"
.venv/bin/abench run "$CONFIG" --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "generation exited with $rc"

# -- 5. free the GPU again -----------------------------------------------------
supervisorctl stop gemma-4-e4b-vllm gemma-4-e2b-vllm >>"$LOG" 2>&1
say "stopped both gemma servers; nothing is running on the GPU"
exit $rc
