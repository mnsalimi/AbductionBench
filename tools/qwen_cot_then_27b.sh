#!/usr/bin/env bash
# Two phases in 20260924-002252_openrouter-trio, generation only (no judges):
#
#   PHASE 1 -- exactly the current Qwen pass: Qwen3.5-4B (:18002) and
#     Qwen3.5-2B (:18001) served side by side with the same settings as
#     tools/qwen_cot_resume.sh, resuming the cot pass with
#     trio_qwen_cot_resume_v2.yaml (native reasoning ON, one answer per
#     request, short queue, 2-hour timeout). Saved answers are reused.
#   PHASE 2 -- every vLLM server down, Qwen3.5-27B (:18007) up alone, then its
#     io pass (native reasoning OFF) and its cot pass (native reasoning ON) on
#     all datasets, treated exactly as the two small Qwens were.
#   END -- servers down, .env files restored, the workbook rebuilt from every
#     record on disk (abench report), and the Drive sidecar stopped: it makes
#     its final VERIFIED upload, and this script waits for it.
#
# The 27B's weights are already in /workspace/.hf_home. If it will not start,
# the only things retried smaller are its parallelism and CUDA graphs -- never
# its 65,536-token window or 32,000-token answer budget.
#
# Start:  bash tools/start_qwen_cot_then_27b.bash      (stops the current run first)
# Watch:  tail -f /workspace/abench_trio.log           (steps tagged [q27])
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
PHASE1_CONFIG=configs/runs/trio_qwen_cot_resume_v2.yaml
IO27_CONFIG=configs/runs/trio_qwen27b_generate_io.yaml
COT27_CONFIG=configs/runs/trio_qwen27b_generate_cot.yaml
BACKUP_SUFFIX=.env.q27-backup
SIDECAR="$REPO/tools/sync_run.sh"

say() { echo "$(date '+%F %T') [q27] $*" | tee -a "$LOG"; }

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
    [ -n "$(tail -c1 "$file")" ] && echo >> "$file"
    if grep -q "^$2=" "$file"; then sed -i "s|^$2=.*|$2=$3|" "$file"; else echo "$2=$3" >> "$file"; fi
}

healthy() { curl -sf -m 10 -H "Authorization: Bearer $ABENCH_API_KEY" "http://127.0.0.1:$1/v1/models" >/dev/null; }

start_service() {  # start_service <service> <port>
    local name=$1 port=$2 waited=0
    supervisorctl start "$name" >>"$LOG" 2>&1
    while [ $waited -lt 2400 ]; do
        if healthy "$port"; then say "$name is up on :$port after ${waited}s"; return 0; fi
        if supervisorctl status "$name" | grep -qE "FATAL|EXITED|BACKOFF|STOPPED"; then
            say "$name failed to start: $(supervisorctl status "$name")"
            tail -8 "/var/log/portal/$name.log" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
            supervisorctl stop "$name" >>"$LOG" 2>&1
            return 1
        fi
        sleep 15; waited=$((waited + 15))
    done
    say "$name did not answer within 40 minutes"; supervisorctl stop "$name" >>"$LOG" 2>&1; return 1
}

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

free_gpu() {
    local running
    running=$(supervisorctl status | awk '/-vllm/ && /RUNNING|STARTING/ {print $1}')
    for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
    say "stopped: ${running:-nothing was running}"
    for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
    say "GPU memory in use: $(gpu_used) MiB"
}

native_reasoning() {  # native_reasoning <port> <model> <true|false>  -> "on" / "off" / "inline" / "no reply"
    local reply
    reply=$(curl -s -m 900 "http://127.0.0.1:$1/v1/chat/completions" \
        -H "Authorization: Bearer $ABENCH_API_KEY" -H "Content-Type: application/json" \
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"A glass of water left in a warm room is empty after three days, and nobody drank from it. Which is more likely: 1) evaporation 2) a leak? Explain briefly, then give the label.\"}],\"max_tokens\":3000,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":$3}}")
    python3 - "$reply" <<'PYEOF'
import json, sys
try:
    msg = json.loads(sys.argv[1])["choices"][0]["message"]
except Exception:
    print("no reply"); sys.exit(0)
native = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
content = (msg.get("content") or "").strip()
inline = "<think>" in content or "</think>" in content or "Thinking Process" in content
if native and not content:
    print("no content")   # the gemma failure: everything swallowed into reasoning
else:
    print("on" if native else ("inline" if inline else "off"))
PYEOF
}

probe() {  # probe <port> <model>  -- the thinking switch must work both ways
    local off on
    off=$(native_reasoning "$1" "$2" false); on=$(native_reasoning "$1" "$2" true)
    if [ "$off" = "off" ] && [ "$on" = "on" ]; then
        say "probe $2: enable_thinking=false -> no reasoning, true -> native reasoning in its own field, content kept -- ok"
        return 0
    fi
    say "probe $2: FAILED -- false gave '$off', true gave '$on' (expected off / on)"
    return 1
}

run_pass() {  # run_pass <label> <config>
    say "$1: abench run $2 --resume $RUN"
    .venv/bin/abench run "$2" --resume "$RUN" >>"$LOG" 2>&1
    local rc=$?
    say "$1 exited with $rc"
    return $rc
}

# -- 0. never two runs in one folder ------------------------------------------
while pgrep -f '[b]in/abench run' >/dev/null; do
    say "another abench run is still going; waiting"; sleep 60
done

# -- Drive: the sidecar backs the folder up from its own process ---------------
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    say "Drive sync sidecar already running"
else
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    say "started the Drive sync sidecar (every 15 minutes; log /tmp/abench_sync.log)"
fi

# ============================ PHASE 1 =========================================
say "PHASE 1: Qwen3.5-2B / 4B cot pass, as it was running"
free_gpu
supervisorctl reread >>"$LOG" 2>&1; supervisorctl update >>"$LOG" 2>&1
for m in qwen3.5-2b qwen3.5-4b; do
    set_env $m MAX_MODEL_LEN 65536
    set_env $m MAX_NUM_SEQS 512
    set_env $m MAX_NUM_BATCHED_TOKENS 16384
    set_env $m ENABLE_PREFIX_CACHING 1
    set_env $m ENFORCE_EAGER 0
    set_env $m REASONING_PARSER qwen3
done
set_env qwen3.5-4b GPU_MEMORY_UTILIZATION 0.52
set_env qwen3.5-2b GPU_MEMORY_UTILIZATION 0.36
phase1_ok=1
for s in qwen3.5-4b-vllm:18002:Qwen/Qwen3.5-4B qwen3.5-2b-vllm:18001:Qwen/Qwen3.5-2B; do
    name=${s%%:*}; rest=${s#*:}; port=${rest%%:*}; model=${rest#*:}
    if ! start_service "$name" "$port" || ! probe "$port" "$model"; then phase1_ok=0; break; fi
done
if [ $phase1_ok = 1 ]; then
    run_pass "phase 1 cot pass (Qwen 2B/4B)" "$PHASE1_CONFIG"
else
    say "PHASE 1 SKIPPED: the small Qwens would not start or failed the probe -- going on to phase 2"
fi

# ============================ PHASE 2 =========================================
say "PHASE 2: every vLLM server down, Qwen3.5-27B up alone"
free_gpu
restore_envs
M=qwen3.5-27b; NAME=qwen3.5-27b-vllm; PORT=18007; MODEL=Qwen/Qwen3.5-27B
set_env $M MAX_MODEL_LEN 65536
set_env $M ENABLE_PREFIX_CACHING 1
set_env $M REASONING_PARSER qwen3
set_env $M MAX_NUM_BATCHED_TOKENS 16384
set_env $M LANGUAGE_MODEL_ONLY 1
# Tried in order; only parallelism, memory share and CUDA graphs change --
# the window stays 65,536.
up=0
for attempt in "128 0.92 0" "96 0.92 0" "64 0.90 0" "64 0.90 1" "32 0.88 1"; do
    read -r seqs util eager <<<"$attempt"
    set_env $M MAX_NUM_SEQS "$seqs"
    set_env $M GPU_MEMORY_UTILIZATION "$util"
    set_env $M ENFORCE_EAGER "$eager"
    say "starting $NAME: MAX_NUM_SEQS=$seqs GPU_MEMORY_UTILIZATION=$util ENFORCE_EAGER=$eager MAX_MODEL_LEN=65536"
    if start_service "$NAME" "$PORT"; then up=1; break; fi
    sleep 20
done

if [ $up = 1 ] && probe "$PORT" "$MODEL"; then
    run_pass "phase 2 io pass (27B, thinking off)" "$IO27_CONFIG"
    run_pass "phase 2 cot pass (27B, thinking on)" "$COT27_CONFIG"
else
    say "PHASE 2 NOT RUN: Qwen3.5-27B did not start or failed the probe (see /var/log/portal/$NAME.log)"
fi

# ============================ END =============================================
free_gpu
restore_envs
say "rebuilding the workbook from every record on disk"
.venv/bin/abench report "runs/$RUN" >>"$LOG" 2>&1 && say "workbook rebuilt" || say "abench report failed (see log)"

say "Drive: final sync"
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    pkill -TERM -f "tools/sync_run.sh .*$RUN"
else
    # No sidecar left: start one and stop it at once -- stopping is what makes
    # it do its final verified pass.
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    sleep 30; pkill -TERM -f "tools/sync_run.sh .*$RUN"
fi
for _ in $(seq 1 720); do pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null || break; sleep 10; done
if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    say "Drive sync still going after 2 hours; left running (log /tmp/abench_sync.log)"
else
    say "Drive sync finished (log /tmp/abench_sync.log). All done."
fi
