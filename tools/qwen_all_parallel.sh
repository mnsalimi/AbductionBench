#!/usr/bin/env bash
# All three Qwens AT ONCE, generation only (no judges), into
# 20260924-002252_openrouter-trio:
#
#   1. Qwen3.5-2B (:18001) and 4B (:18002) on this box's vLLM: reused if they
#      are already up, otherwise started with the settings they have run with
#      all along (512 sequences, 65,536 window, qwen3 reasoning parser).
#      Both are probed: thinking off -> no reasoning, on -> reasoning apart.
#   2. Qwen3.5-27B through OpenRouter, cheapest host only (Alibaba): its last
#      io answers first, reasoning OFF (probed), a pass of a few minutes.
#   3. THE COT PASS FOR ALL THREE IN ONE RUN (trio_all_qwen_cot_parallel.yaml),
#      reasoning ON: 256 + 128 local requests in flight, 512 to OpenRouter.
#      One run, so one writer to the folder.
#   4. the same pass resumed once more: only answers that ended as errors
#      (rate-limit retries used up, 402) are asked again.
#   5. servers down, workbook rebuilt, final verified Drive sync.
#
#   Out of OpenRouter credit (HTTP 402) -> OPENROUTER_API_KEY_BACKUP from
#   /workspace/.env takes over and the pass is resumed.
#
# Start:  bash tools/start_qwen_all_parallel.bash   (stops the current run first)
# Watch:  tail -f /workspace/abench_trio.log | grep allq
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
ALL_CONFIG=configs/runs/trio_all_qwen_cot_parallel.yaml
QWENS_ONLY_CONFIG=configs/runs/trio_qwen_cot_resume_v2.yaml
IO27_CONFIG=configs/runs/trio_qwen27b_or_io_a.yaml
COT27_CONFIG=configs/runs/trio_qwen27b_or_cot.yaml
BACKUP_SUFFIX=.env.allq-backup
SIDECAR="$REPO/tools/sync_run.sh"

say() { echo "$(date '+%F %T') [allq] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "runs/$RUN/datasets" ] || { say "no run folder runs/$RUN"; exit 1; }
if [ -z "${OPENROUTER_API_KEY:-}" ]; then set -a; . /workspace/.env; set +a; fi
ABENCH_API_KEY=$(set -a; . "$SERVING/qwen3.5-2b/.env"; echo "${VLLM_API_KEY:-}")
export ABENCH_API_KEY OPENROUTER_API_KEY
[ -n "$ABENCH_API_KEY" ] || { say "no VLLM_API_KEY in $SERVING/qwen3.5-2b/.env"; exit 1; }
# The second OpenRouter key, used only once the first runs out of credit (402).
OPENROUTER_API_KEY_BACKUP=$(set -a; . /workspace/.env; echo "${OPENROUTER_API_KEY_BACKUP:-}")
KEY_SWITCHED=0

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
        -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"A glass of water left in a warm room is empty after three days, and nobody drank from it. Which is more likely: 1) evaporation 2) a leak? Explain briefly, then give the label.\"}],\"max_tokens\":16000,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":$3}}")
    python3 - "$reply" <<'PYEOF'
import json, sys
try:
    msg = json.loads(sys.argv[1])["choices"][0]["message"]
except Exception:
    print("no reply"); sys.exit(0)
native = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
content = (msg.get("content") or "").strip()
inline = "<think>" in content or "</think>" in content or "Thinking Process" in content
finish = json.loads(sys.argv[1])["choices"][0].get("finish_reason")
if native and not content and finish != "length":
    print("no content")   # the gemma failure: a FINISHED reply swallowed into reasoning
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

switch_key() {  # to the backup OpenRouter key, once; false if there is none left
    if [ $KEY_SWITCHED = 1 ] || [ -z "$OPENROUTER_API_KEY_BACKUP" ]; then return 1; fi
    OPENROUTER_API_KEY=$OPENROUTER_API_KEY_BACKUP; export OPENROUTER_API_KEY; KEY_SWITCHED=1
    say "OpenRouter key out of credit (HTTP 402): switched to the backup key"
}

probe_thinking() {  # probe_thinking <config> on|off
    local out
    out=$(.venv/bin/python tools/probe_openrouter_thinking.py "$1" "$2" 2>&1 | tail -1)
    if [[ "$out" == *"402"* ]] && switch_key; then
        out=$(.venv/bin/python tools/probe_openrouter_thinking.py "$1" "$2" 2>&1 | tail -1)
    fi
    say "probe $(basename "$1") (want $2): $out"
    [[ "$out" == "$2 ("* ]]
}

paid_pass() {  # paid_pass <label> <config> -- an OpenRouter pass; on 402, again with the backup key
    local start n402
    start=$(stat -c %s "$LOG")
    run_pass "$1" "$2"
    n402=$(tail -c +$((start + 1)) "$LOG" | grep -c "HTTP 402")
    if [ "$n402" -gt 0 ]; then
        say "$1: $n402 log line(s) with HTTP 402 (out of credit)"
        if switch_key; then
            start=$(stat -c %s "$LOG")
            run_pass "$1 -- again with the backup key (only the failed answers are asked)" "$2"
            n402=$(tail -c +$((start + 1)) "$LOG" | grep -c "HTTP 402")
            [ "$n402" -gt 0 ] && say "$1: the backup key ran out of credit too ($n402 x HTTP 402) -- top one up and resume"
        else
            say "$1: no backup key left -- top up and resume"
        fi
    fi
}



# -- 0. one run at a time in this folder ---------------------------------------
while pgrep -f 'tools/[a]fter_27b_phase1\.sh|tools/[q]wen_cot_then_27b' >/dev/null || pgrep -f '[b]in/abench run' >/dev/null; do
    say "another run or control script is still going; waiting"; sleep 30
done

if pgrep -f "tools/sync_run.sh .*$RUN" >/dev/null; then
    say "Drive sync sidecar already running"
else
    nohup bash "$SIDECAR" "$REPO/runs/$RUN" >> /tmp/abench_sync.log 2>&1 &
    say "started the Drive sync sidecar (every 15 minutes; log /tmp/abench_sync.log)"
fi

# -- 1. the two local Qwens ----------------------------------------------------
qwens_ok=1
if healthy 18001 && healthy 18002; then
    say "Qwen3.5-2B and 4B are already up -- reusing them as they are"
else
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
    for s in qwen3.5-4b-vllm:18002 qwen3.5-2b-vllm:18001; do
        start_service "${s%%:*}" "${s#*:}" || { qwens_ok=0; break; }
    done
fi
if [ $qwens_ok = 1 ]; then
    probe 18002 Qwen/Qwen3.5-4B && probe 18001 Qwen/Qwen3.5-2B || qwens_ok=0
fi
[ $qwens_ok = 1 ] || say "the local Qwens are NOT usable -- they will not be run; the 27B still will"

# -- 2. the 27B's last io answers (reasoning OFF) ------------------------------
if probe_thinking "$IO27_CONFIG" off; then
    paid_pass "27B io remainder (OpenRouter, thinking off)" "$IO27_CONFIG"
else
    say "27B io NOT RUN: the probe did not come back with reasoning off"
fi

# -- 3 + 4. cot for all three at once, then once more for errors ---------------
cot27_ok=0
if probe_thinking "$COT27_CONFIG" on || { sleep 60; probe_thinking "$COT27_CONFIG" on; }; then cot27_ok=1; fi
if [ $qwens_ok = 1 ] && [ $cot27_ok = 1 ]; then
    COT_CONFIG=$ALL_CONFIG; what="all three Qwens"
elif [ $qwens_ok = 1 ]; then
    COT_CONFIG=$QWENS_ONLY_CONFIG; what="Qwen 2B/4B only (the 27B failed its probe)"
elif [ $cot27_ok = 1 ]; then
    COT_CONFIG=$COT27_CONFIG; what="the 27B only (the local Qwens are down)"
else
    COT_CONFIG=""; say "NO COT PASS: neither the local Qwens nor the 27B are usable"
fi
if [ -n "$COT_CONFIG" ]; then
    paid_pass "cot pass, $what" "$COT_CONFIG"
    paid_pass "cot retry pass, $what (only answers that ended as errors)" "$COT_CONFIG"
fi

# -- 5. the end ------------------------------------------------------------------
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
