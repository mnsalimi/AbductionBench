#!/usr/bin/env bash
# Two phases in 20260924-002252_openrouter-trio, generation only (no judges):
#
#   PHASE 1 -- exactly the current Qwen pass: Qwen3.5-4B (:18002) and
#     Qwen3.5-2B (:18001) served side by side with the settings of
#     tools/qwen_cot_resume.sh, resuming the cot pass with
#     trio_qwen_cot_resume_v2.yaml (native reasoning ON). Saved answers reused.
#     Then every vLLM server is stopped -- nothing else needs the GPU.
#   PHASE 2 -- Qwen3.5-27B through OpenRouter (cheapest provider only, 256
#     requests in flight) on all datasets, treated as the small Qwens were:
#       io  -- native reasoning OFF. The providers do not all honour the same
#              switch (Alibaba ignored reasoning.enabled=false on 2026-09-25),
#              so the io variants a-d are probed in order, one short request
#              each, and the first that returns NO reasoning is run. If none
#              does, io is NOT run with reasoning on -- it is skipped and said.
#       cot -- native reasoning ON, probed the same way first.
#   END -- the workbook rebuilt from every record on disk (abench report), then
#     the Drive sidecar is stopped: it makes its final VERIFIED upload, and
#     this script waits for it.
#
# Start:  bash tools/start_qwen_cot_then_27b.bash      (stops the current run first)
# Watch:  tail -f /workspace/abench_trio.log           (steps tagged [q27or])
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
PHASE1_CONFIG=configs/runs/trio_qwen_cot_resume_v2.yaml
COT27_CONFIG=configs/runs/trio_qwen27b_or_cot.yaml
IO27_VARIANTS="a b c d"
BACKUP_SUFFIX=.env.q27-backup
SIDECAR="$REPO/tools/sync_run.sh"

say() { echo "$(date '+%F %T') [q27or] $*" | tee -a "$LOG"; }

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

probe_thinking() {  # probe_thinking <config> on|off
    local out rc
    out=$(.venv/bin/python tools/probe_openrouter_thinking.py "$1" "$2" 2>&1 | tail -1); rc=$?
    say "probe $(basename "$1") (want $2): $out"
    [[ "$out" == "$2 ("* ]]
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


# -- after phase 1: nothing needs the GPU any more ------------------------------
free_gpu
restore_envs

# ============================ PHASE 2 =========================================
say "PHASE 2: Qwen3.5-27B through OpenRouter, io (thinking off) then cot (thinking on)"
io_config=""
for round in 1 2; do
    for v in $IO27_VARIANTS; do
        if probe_thinking "configs/runs/trio_qwen27b_or_io_$v.yaml" off; then
            io_config="configs/runs/trio_qwen27b_or_io_$v.yaml"; break 2
        fi
    done
    [ $round = 1 ] && sleep 60
done
if [ -n "$io_config" ]; then
    run_pass "phase 2 io pass (27B via OpenRouter, thinking off, $(basename "$io_config"))" "$io_config"
else
    say "PHASE 2 IO NOT RUN: no variant turned the 27B's native reasoning off on OpenRouter -- nothing generated with reasoning on"
fi
if probe_thinking "$COT27_CONFIG" on || { sleep 60; probe_thinking "$COT27_CONFIG" on; }; then
    run_pass "phase 2 cot pass (27B via OpenRouter, thinking on)" "$COT27_CONFIG"
else
    say "PHASE 2 COT NOT RUN: the 27B did not return native reasoning with thinking on"
fi

# ============================ END =============================================
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
