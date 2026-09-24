#!/usr/bin/env bash
# Everything left for 20260924-002252_openrouter-trio, unattended, in order.
# Every pass resumes into that ONE run folder and backs up to the same Drive
# folder (engine.sync), so it ends as one workbook with every model in it.
#
#   STAGE 1  judge: gpt-oss-120b served locally (already up)
#            tools/resume_openrouter_trio.sh --
#              io again for luna (reasoning off) and gemini (effort low) on every
#              dataset, gemma's io on the datasets it has not reached, then
#              every remaining cot task for all three. Every finished cot
#              sample is re-judged for its reasoning metrics on the way (steps
#              prompt v4, one reasoning source per reply).
#   STAGE 2  judge: openai/gpt-oss-120b on OpenRouter; GPU: gemma-4-E4B,
#            gemma-4-E2B and Qwen3.5-2B side by side -- io (thinking off), then cot.
#   STAGE 3  judge: the same; GPU: Qwen3.5-27B alone -- io (thinking off), then cot.
#
# A stage that cannot start its servers is skipped and the next one is tried;
# only an interrupt (exit 130) stops the whole script. The servers' .env files
# are changed for the stage that needs it and put back afterwards, even if the
# script is killed.
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/overnight_openrouter_trio.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log        (script steps are tagged [overnight])
# Stop:   pkill -f overnight_openrouter_trio.sh; pkill -f resume_openrouter_trio.sh; pkill -INT -f 'bin/abench run'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
ENV_BACKUP_SUFFIX=.env.overnight-backup

say() { echo "$(date '+%F %T') [overnight] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN -- nothing to resume"; exit 1; }

# -- keys ---------------------------------------------------------------------
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
# Every local server shares one VLLM_API_KEY (vllm_serving/README.md).
ABENCH_API_KEY=$(set -a; . "$SERVING/gpt-oss-120b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "${OPENROUTER_API_KEY:-}" ] && [ -n "$ABENCH_API_KEY" ] || { say "missing OPENROUTER_API_KEY or VLLM_API_KEY"; exit 1; }

# -- server .env edits, always undone -----------------------------------------
restore_envs() {
    for backup in "$SERVING"/*/"$ENV_BACKUP_SUFFIX"; do
        [ -f "$backup" ] || continue
        mv -f "$backup" "$(dirname "$backup")/.env"
        say "restored $(dirname "$backup")/.env"
    done
}
trap restore_envs EXIT

set_env() {  # set_env <model dir> KEY VALUE  (backs the file up once per stage)
    local file="$SERVING/$1/.env"
    [ -f "$SERVING/$1/$ENV_BACKUP_SUFFIX" ] || cp -p "$file" "$SERVING/$1/$ENV_BACKUP_SUFFIX"
    if grep -q "^$2=" "$file"; then
        sed -i "s|^$2=.*|$2=$3|" "$file"
    else
        echo "$2=$3" >> "$file"
    fi
}

# -- services -----------------------------------------------------------------
healthy() {  # healthy <port>
    curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null
}

start_service() {  # start_service <service> <port>  -> 0 once it answers
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
    say "$name did not answer within 30 minutes"
    supervisorctl stop "$name" >>"$LOG" 2>&1
    return 1
}

stop_services() {
    for name in "$@"; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
    local waited=0
    # The next stage's servers check free memory at startup.
    while [ $waited -lt 300 ]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "${used:-99999}" -lt 4000 ] && break
        sleep 10; waited=$((waited + 10))
    done
    say "stopped $* -- GPU memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
}

thinking_is_off() {  # thinking_is_off <port> <model name>  -> 0 if io really gets no chain
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
thought = "<think>" in content or "Thinking Process" in content or native.strip()
print(repr(content[:80]))
sys.exit(1 if thought else 0)
EOF
}

wait_idle() {
    while pgrep -f 'bin/abench run' >/dev/null || pgrep -f "rclone copy .*$RUN" >/dev/null; do sleep 60; done
}

run_pass() {  # run_pass <label> <config>  -> the pass's exit code; 130 ends the script
    wait_idle
    say "$1: abench run $2 --resume $RUN"
    .venv/bin/abench run "$2" --resume "$RUN" >>"$LOG" 2>&1
    local rc=$?
    say "$1 exited with $rc"
    if [ $rc -eq 130 ]; then say "interrupted -- stopping here; re-run this script to continue"; exit 130; fi
    return $rc
}

# ============================================================================
# STAGE 1 -- the trio, judged by the local gpt-oss-120b
# ============================================================================
say "STAGE 1/3: luna+gemini io again, the trio's remaining work, reasoning metrics recomputed"
wait_idle
if ! healthy 18004; then
    say "the local judge is down; starting gpt-oss-120b-vllm"
    start_service gpt-oss-120b-vllm 18004 || say "WARNING: no local judge -- stage 1 will fail its endpoint check"
fi
bash tools/resume_openrouter_trio.sh
rc=$?
say "stage 1 finished with $rc"
[ $rc -eq 130 ] && { say "interrupted -- stopping here"; exit 130; }

# ============================================================================
# STAGE 2 -- gemma-4-E4B + gemma-4-E2B + Qwen3.5-2B, judged on OpenRouter
# ============================================================================
say "STAGE 2/3: gemma-4-E4B, gemma-4-E2B, Qwen3.5-2B on this GPU; judges on OpenRouter"
wait_idle
stop_services gpt-oss-120b-vllm gpt-oss-120b-tunnel

# 0.32 + 0.22 + 0.28 = 0.82 of the 95.6 GiB card, ~17 GiB left for cuBLAS
# workspace outside vLLM's reservation (0.78 beside another model once died in
# service for want of it). Windows 32,768 for the gemmas (was 16,384, which
# clamped answers); 128 sequences each, the KV pool decides how many run.
set_env gemma-4-e4b GPU_MEMORY_UTILIZATION 0.32; set_env gemma-4-e4b MAX_MODEL_LEN 32768; set_env gemma-4-e4b MAX_NUM_SEQS 128
set_env gemma-4-e2b GPU_MEMORY_UTILIZATION 0.22; set_env gemma-4-e2b MAX_MODEL_LEN 32768; set_env gemma-4-e2b MAX_NUM_SEQS 128
set_env qwen3.5-2b  GPU_MEMORY_UTILIZATION 0.28; set_env qwen3.5-2b  MAX_NUM_SEQS 128

stage2_ok=1
export ABENCH_E4B_WINDOW=32768 ABENCH_E2B_WINDOW=32768
if ! start_service gemma-4-e4b-vllm 18000; then
    say "retrying gemma-4-E4B with its old 16,384 window"
    set_env gemma-4-e4b MAX_MODEL_LEN 16384; export ABENCH_E4B_WINDOW=16384
    start_service gemma-4-e4b-vllm 18000 || stage2_ok=0
fi
if [ $stage2_ok -eq 1 ] && ! start_service gemma-4-e2b-vllm 18005; then
    say "retrying gemma-4-E2B with its old 16,384 window"
    set_env gemma-4-e2b MAX_MODEL_LEN 16384; export ABENCH_E2B_WINDOW=16384
    start_service gemma-4-e2b-vllm 18005 || stage2_ok=0
fi
if [ $stage2_ok -eq 1 ] && ! start_service qwen3.5-2b-vllm 18001; then
    say "retrying Qwen3.5-2B with 64 sequences"
    set_env qwen3.5-2b MAX_NUM_SEQS 64
    start_service qwen3.5-2b-vllm 18001 || stage2_ok=0
fi

if [ $stage2_ok -eq 1 ]; then
    for probe in "18000 google/gemma-4-E4B-it" "18005 google/gemma-4-E2B-it" "18001 Qwen/Qwen3.5-2B"; do
        # shellcheck disable=SC2086
        if thinking_is_off $probe >>"$LOG" 2>&1; then say "io probe ${probe#* }: no thinking -- ok"
        else say "io probe ${probe#* }: THINKING STILL ON -- io for stage 2 skipped"; stage2_ok=2; fi
    done
    [ $stage2_ok -eq 1 ] && run_pass "stage 2 io" configs/runs/trio_stage2_small_local_io.yaml
    run_pass "stage 2 cot" configs/runs/trio_stage2_small_local.yaml
else
    say "STAGE 2 SKIPPED: its servers would not start (see above)"
fi
stop_services gemma-4-e4b-vllm gemma-4-e2b-vllm qwen3.5-2b-vllm
restore_envs

# ============================================================================
# STAGE 3 -- Qwen3.5-27B alone, judged on OpenRouter
# ============================================================================
say "STAGE 3/3: Qwen3.5-27B on this GPU; judges on OpenRouter"
wait_idle
# 0.93 and a 32,768 window as configured; 32 -> 128 sequences (a hybrid
# linear-attention model, so its KV is cheap), falling back to 32.
set_env qwen3.5-27b MAX_NUM_SEQS 128
export ABENCH_27B_BATCHES=16
stage3_ok=1
if ! start_service qwen3.5-27b-vllm 18007; then
    say "retrying Qwen3.5-27B with its old 32 sequences"
    set_env qwen3.5-27b MAX_NUM_SEQS 32; export ABENCH_27B_BATCHES=4
    start_service qwen3.5-27b-vllm 18007 || stage3_ok=0
fi
if [ $stage3_ok -eq 1 ]; then
    if thinking_is_off 18007 Qwen/Qwen3.5-27B >>"$LOG" 2>&1; then
        say "io probe Qwen3.5-27B: no thinking -- ok"
        run_pass "stage 3 io" configs/runs/trio_stage3_qwen27b_io.yaml
    else
        say "io probe Qwen3.5-27B: THINKING STILL ON -- io for stage 3 skipped"
    fi
    run_pass "stage 3 cot" configs/runs/trio_stage3_qwen27b.yaml
else
    say "STAGE 3 SKIPPED: Qwen3.5-27B would not start (see above)"
fi
stop_services qwen3.5-27b-vllm
restore_envs

say "ALL STAGES DONE. Nothing is left running on the GPU. Results: runs/$RUN/reports/abductionbench_results.xlsx (and on Drive)"
