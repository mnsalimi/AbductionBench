#!/usr/bin/env bash
# Every judge that is still owed in 20260924-002252_openrouter-trio, with
# gpt-oss-120b served on THIS box -- nothing goes to OpenRouter, and no model
# under test is asked anything.
#
#   0. waits until no generation is running (tools/qwen_finish.sh and any
#      abench run), then frees the GPU
#   1. CACHE COVERAGE CHECK (read-only): how many of the trio's reasoning-judge
#      calls the cache already holds -> runs/<run>/reasoning_cache_coverage.txt
#   2. serves gpt-oss-120b alone (:18004): 0.90 of the card, 65,536-token
#      window, 256 sequences, 16,384-token scheduler step; 32 calls x 8 = 256
#      in flight from the judges -- the KV cache, not a cap, is what limits it
#   3. TRIO: old reasoning metrics cleared (the cache is KEPT; hypobench left
#      as it is), then re-judged -- every cached verdict read back for free,
#      anchoring_point (v3@1.1) and any other miss asked
#   4. FIVE NEW MODELS (gemma-4-E4B/E2B, Qwen3.5-2B/4B, Qwen3.5-27B):
#      a. answer judge on io and cot   (tools/judge_answers_offline.py)
#      b. reasoning judge on cot       (abench judge-reasoning)
#      Only replies that exist are judged: the 27B's cot samples stored as
#      errors are skipped, and replies that never gave an answer are scored as
#      unreadable, not judged.
#   5. judge server stopped, workbook rebuilt ONCE, final verified Drive sync
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/judge_all_after_generation.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log | grep judge-all
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
TRIO_CONFIG=configs/runs/trio_judge_trio.yaml
NEW_CONFIG=configs/runs/trio_judge_new.yaml
NEW_MODELS=gemma-4-e4b-local,gemma-4-e2b-local,qwen3-5-2b-local,qwen3-5-4b-local,qwen3-5-27b-openrouter
JUDGE_SERVICE=gpt-oss-120b-vllm
JUDGE_PORT=18004
BACKUP_SUFFIX=.env.judge-all-backup
SIDECAR="$REPO/tools/sync_run.sh"

say() { echo "$(date '+%F %T') [judge-all] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN"; exit 1; }
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . "$SERVING/gpt-oss-120b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/gpt-oss-120b/.env"; exit 1; }

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
    [ -n "$(tail -c1 "$file")" ] && echo >> "$file"
    if grep -q "^$2=" "$file"; then sed -i "s|^$2=.*|$2=$3|" "$file"; else echo "$2=$3" >> "$file"; fi
}

healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

free_gpu() {
    local running
    running=$(supervisorctl status | awk '/-vllm/ && /RUNNING|STARTING/ {print $1}')
    for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
    say "stopped: ${running:-nothing was running}"
    for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
    say "GPU memory in use: $(gpu_used) MiB"
}

step() {  # step <label> <command...> -- runs it, logs how it ended, never aborts the script
    local label=$1; shift
    say "$label: $*"
    "$@" >>"$LOG" 2>&1
    local rc=$?
    say "$label exited with $rc"
    return $rc
}

# -- 0. generation first ---------------------------------------------------------
while pgrep -f 'tools/[q]wen_finish\.sh|tools/[q]wen_all_parallel\.sh' >/dev/null || pgrep -f '[b]in/abench run' >/dev/null; do
    say "generation is still running; waiting"; sleep 60
done
free_gpu
# Backed up to Drive from its own process all along -- this runs for hours.
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    say "Drive sync sidecar already running"
else
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    say "started the Drive sync sidecar (every 15 minutes; log /tmp/abench_sync.log)"
fi

# -- 1. cache coverage (read-only) -------------------------------------------------
step "1. cache coverage check (trio, read-only)" \
    .venv/bin/python tools/reasoning_cache_coverage.py "runs/$RUN" \
    --config "$TRIO_CONFIG" --out "runs/$RUN/reasoning_cache_coverage.txt"
[ -f "runs/$RUN/reasoning_cache_coverage.txt" ] && sed 's/^/    /' "runs/$RUN/reasoning_cache_coverage.txt" | tee -a "$LOG" >/dev/null

# -- 2. the judge server ------------------------------------------------------------
set_env gpt-oss-120b GPU_MEMORY_UTILIZATION 0.90
set_env gpt-oss-120b MAX_MODEL_LEN 65536
set_env gpt-oss-120b MAX_NUM_SEQS 256
set_env gpt-oss-120b MAX_NUM_BATCHED_TOKENS 16384
set_env gpt-oss-120b ENFORCE_EAGER 0
supervisorctl reread >>"$LOG" 2>&1; supervisorctl update >>"$LOG" 2>&1
supervisorctl start "$JUDGE_SERVICE" >>"$LOG" 2>&1
up=0
for waited in $(seq 0 15 2400); do
    if healthy $JUDGE_PORT; then up=1; say "$JUDGE_SERVICE is up on :$JUDGE_PORT after ${waited}s"; break; fi
    if supervisorctl status "$JUDGE_SERVICE" | grep -qE "FATAL|EXITED|BACKOFF"; then
        say "$JUDGE_SERVICE failed to start: $(supervisorctl status "$JUDGE_SERVICE")"
        tail -8 "/var/log/portal/$JUDGE_SERVICE.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
        break
    fi
    sleep 15
done
if [ $up = 1 ]; then
    reply=$(curl -s -m 300 "http://127.0.0.1:$JUDGE_PORT/v1/chat/completions" \
        -H "Authorization: Bearer $ABENCH_API_KEY" -H "Content-Type: application/json" \
        -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":256,"temperature":0,"reasoning_effort":"low"}')
    if echo "$reply" | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; sys.exit(0 if (m.get("content") or "").strip() else 1)' 2>/dev/null; then
        say "judge probe: answered -- ok"
    else
        say "judge probe FAILED: ${reply:0:300}"; up=0
    fi
fi
if [ $up != 1 ]; then
    say "NO JUDGE -- nothing judged, nothing cleared. Fix $JUDGE_SERVICE and run this again."
    supervisorctl stop "$JUDGE_SERVICE" >>"$LOG" 2>&1
    exit 1
fi

# -- 3. the trio: clear (cache kept), then re-judge ------------------------------
if step "3a. clear the old reasoning metrics (cache kept, hypobench untouched)" \
        .venv/bin/python tools/clear_reasoning_metrics.py "runs/$RUN" --apply --keep-cache --exclude hypobench; then
    :
else
    say "some records.jsonl were written in the last 2 minutes and were skipped; clearing again in 3 minutes"
    sleep 180
    step "3a. clear again" .venv/bin/python tools/clear_reasoning_metrics.py "runs/$RUN" --apply --keep-cache --exclude hypobench
fi
step "3b. trio reasoning metrics (cache + anchoring + misses)" \
    .venv/bin/abench judge-reasoning "runs/$RUN" --config "$TRIO_CONFIG" --no-report

# -- 4. the five new models ----------------------------------------------------------
step "4a. answer judge, five new models (io + cot)" \
    .venv/bin/python tools/judge_answers_offline.py "runs/$RUN" --config "$NEW_CONFIG" --models "$NEW_MODELS"
step "4b. reasoning metrics, five new models (cot)" \
    .venv/bin/abench judge-reasoning "runs/$RUN" --config "$NEW_CONFIG" --no-report

# -- 5. the end ----------------------------------------------------------------------
supervisorctl stop "$JUDGE_SERVICE" >>"$LOG" 2>&1
say "judge server stopped; the GPU is free"
restore_envs
step "5. workbook, rebuilt once from every record on disk" .venv/bin/abench report "runs/$RUN"

say "Drive: final sync"
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    pkill -TERM -f "tools/sync_run.sh .*$RUN"
else
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    sleep 30; pkill -TERM -f "tools/sync_run.sh .*$RUN"
fi
for _ in $(seq 1 720); do pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null || break; sleep 10; done
pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null \
    && say "Drive sync still going after 2 hours (log /tmp/abench_sync.log)" \
    || say "Drive sync finished. All done."
