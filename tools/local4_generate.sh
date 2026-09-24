#!/usr/bin/env bash
# gemma-4-E4B, gemma-4-E2B, Qwen3.5-2B and Qwen3.5-4B -- GENERATION ONLY, again,
# into 20260924-002252_openrouter-trio, with the native reasoning switch set
# per mode: io OFF, cot ON (enable_thinking false / true), for all four.
#
#   1. once: sets the gemmas' earlier task folders aside (their cot ran with
#      native reasoning off) so every question is asked again
#   2. stops every vLLM server, then serves all four side by side with vLLM's
#      reasoning parsers, so native reasoning arrives in its own `reasoning`
#      field (the workbook's reasoning column, as for luna and gemini)
#   3. PROVES the switch on each model before generating anything: with
#      enable_thinking=false it must return no reasoning, with true it must
#      return some. Any model that fails stops the whole script -- nothing is
#      generated, and the log says which model and which way.
#   4. io pass, then cot pass: every dataset, no judge, synced to Drive
#   5. stops all four servers and restores their .env files
#
# Start:  cd /workspace/AbductionBench && nohup bash tools/local4_generate.sh > /dev/null 2>&1 &
# Watch:  tail -f /workspace/abench_trio.log      (steps tagged [local4])
# Stop:   pkill -f local4_generate.sh; pkill -INT -f 'bin/abench run'
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
SERVING=/workspace/vllm_serving
LOG=/workspace/abench_trio.log
CONFIG_IO=configs/runs/trio_local4_generate_io.yaml
CONFIG_COT=configs/runs/trio_local4_generate_cot.yaml
SETASIDE=$REPO/runs/_superseded/${RUN}__local4_before_native_reasoning
BACKUP_SUFFIX=.env.local4-backup

say() { echo "$(date '+%F %T') [local4] $*" | tee -a "$LOG"; }

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
while pgrep -f 'bin/abench run' >/dev/null || pgrep -f "rclone copy .*$RUN" >/dev/null; do
    say "another abench run or its upload is still going; waiting"; sleep 60
done

# -- 1. set the earlier gemma answers aside, once ------------------------------
if [ -e "$SETASIDE/.moved_complete" ]; then
    say "earlier answers already set aside ($SETASIDE)"
else
    moved=0
    for model in gemma-4-e4b-local gemma-4-e2b-local qwen3-5-2b-local qwen3-5-4b-local; do
        for task in "runs/$RUN"/datasets/*/"$model"/*; do
            [ -d "$task" ] || continue
            rel=${task#"runs/$RUN"/}
            mkdir -p "$SETASIDE/$(dirname "$rel")"
            [ -e "$SETASIDE/$rel" ] && { say "refusing to overwrite $SETASIDE/$rel"; exit 1; }
            mv "$task" "$SETASIDE/$rel" || { say "could not move $rel"; exit 1; }
            moved=$((moved + 1))
        done
    done
    touch "$SETASIDE/.moved_complete"
    say "set aside $moved earlier task folder(s) of the four models -> $SETASIDE"
fi

# -- 2. free the GPU, then serve all four --------------------------------------
running=$(supervisorctl status | awk '/-vllm/ && /RUNNING/ {print $1}')
for name in $running; do supervisorctl stop "$name" >>"$LOG" 2>&1; done
say "stopped: ${running:-nothing was running}"
for _ in $(seq 1 30); do [ "$(gpu_used)" -lt 4000 ] && break; sleep 10; done
say "GPU memory in use: $(gpu_used) MiB"
supervisorctl reread >>"$LOG" 2>&1; supervisorctl update qwen3.5-4b-vllm >>"$LOG" 2>&1

# 0.26 + 0.19 + 0.24 + 0.17 = 0.86 of 95.6 GiB, ~13 GiB left for the four
# processes' CUDA contexts and cuBLAS workspace outside vLLM's reservations.
# 512 sequences each (the KV pool decides how many decode), 16,384-token
# scheduler steps, 65,536-token windows: every dataset keeps its full
# 32,000-token answer budget. No fallback to anything smaller.
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

SERVERS="gemma-4-e4b-vllm:18000:google/gemma-4-E4B-it gemma-4-e2b-vllm:18005:google/gemma-4-E2B-it qwen3.5-4b-vllm:18002:Qwen/Qwen3.5-4B qwen3.5-2b-vllm:18001:Qwen/Qwen3.5-2B"
stop_all() { for s in $SERVERS; do supervisorctl stop "${s%%:*}" >>"$LOG" 2>&1; done; }
for s in $SERVERS; do
    name=${s%%:*}; rest=${s#*:}; port=${rest%%:*}
    start_service "$name" "$port" || { say "$name will not start with these settings -- stopping, nothing generated"; stop_all; exit 1; }
done

# -- 3. prove the native reasoning switch on every model -----------------------
failed=0
for s in $SERVERS; do
    rest=${s#*:}; port=${rest%%:*}; model=${rest#*:}
    off=$(native_reasoning "$port" "$model" false); on=$(native_reasoning "$port" "$model" true)
    if [ "$off" = "off" ] && [ "$on" = "on" ]; then
        say "probe $model: enable_thinking=false -> no reasoning, true -> native reasoning in its own field -- ok"
    else
        say "probe $model: FAILED -- enable_thinking=false gave '$off', true gave '$on' (expected off / on)"
        failed=1
    fi
done
if [ $failed -ne 0 ]; then
    say "NOT GENERATING: at least one model does not switch native reasoning as required. Nothing was run; decide how to handle it first."
    stop_all
    exit 2
fi

# -- 4. generate: io (native reasoning off), then cot (native reasoning on) -----
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
stop_all
say "stopped all four servers; nothing is running on the GPU"
exit $rc
