#!/usr/bin/env bash
# Finish the openrouter-trio run (20260924-002252_openrouter-trio) in two passes
# into the SAME run folder, backed up to the SAME Drive folder:
#
#   pass 1  io, every enabled dataset (configs/runs/openrouter_trio_io.yaml):
#             gpt-5.6-luna      reasoning {enabled: false}   -- verified 0 reasoning tokens
#             gemini-3.8-flash  reasoning {effort: low}      -- the API refuses "off"
#             gemma-4-31b-it    unchanged (it never reasoned in io)
#   pass 2  cot, every enabled dataset, settings exactly as the run started
#           (gemini keeps its provider-default reasoning effort there)
#
# luna's and gemini's io was first answered with reasoning ON by mistake. Their
# io task folders are moved out of the run before pass 1, so pass 1 asks every
# one of those questions again rather than reusing the old answers (the
# reasoning setting is not part of the resume fingerprint, so it would reuse
# them otherwise). Their copies on Drive go to Drive's trash. Nothing else is
# touched: cot records, gemma's io records, the caches and the logs stay, and
# every finished answer in them is reused.
#
# Safe to start again after a stop: the move happens once (a marker records it),
# and each pass resumes from its checkpoints.
#
# Stop it:  pkill -INT -f 'bin/abench run'   (drains in-flight work; this script
#           then exits without starting the next pass)
set -uo pipefail

RUN=20260924-002252_openrouter-trio
REPO=/workspace/AbductionBench
RUN_DIR=$REPO/runs/$RUN
BACKUP=$REPO/runs/_superseded/${RUN}__io_reasoning_on
MARKER=$BACKUP/.moved_complete
REMOTE=gdrive:AbductionBench/$RUN
LOG=/workspace/abench_trio.log
REDONE_MODELS=(gpt-5.6-luna-openrouter gemini-3.8-flash-openrouter)

say() { echo "$(date '+%F %T') [resume-trio] $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1
[ -d "$RUN_DIR/datasets" ] || { say "no run folder at $RUN_DIR -- nothing to resume"; exit 1; }

# -- keys ---------------------------------------------------------------------
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    set -a; . /workspace/.env; set +a
fi
if [ -z "${ABENCH_API_KEY:-}" ]; then
    ABENCH_API_KEY=$(set -a; . /workspace/vllm_serving/gpt-oss-120b/.env; echo "${VLLM_API_KEY:-}")
    export ABENCH_API_KEY
fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { say "OPENROUTER_API_KEY is not set"; exit 1; }
[ -n "${ABENCH_API_KEY:-}" ] || { say "ABENCH_API_KEY is not set"; exit 1; }

# -- never two runs in one folder ---------------------------------------------
# The stopped run may still be making its last upload; both passes would share
# its folder and its sync staging directory.
while pgrep -f 'bin/abench run' >/dev/null || pgrep -f "rclone copy .*$RUN" >/dev/null; do
    say "an earlier abench run or its upload is still going; waiting"
    sleep 60
done

# -- the judge has to be up ---------------------------------------------------
if ! curl -sf -m 20 -H "Authorization: Bearer $ABENCH_API_KEY" http://127.0.0.1:18004/v1/models >/dev/null; then
    say "the judge (gpt-oss-120b on :18004) is not answering -- start it: supervisorctl start gpt-oss-120b-vllm"
    exit 1
fi

# -- set the reasoning-on io aside, once --------------------------------------
if [ -e "$MARKER" ]; then
    say "reasoning-on io was already set aside ($BACKUP); not moving anything"
else
    moved=0
    for model in "${REDONE_MODELS[@]}"; do
        for task in "$RUN_DIR"/datasets/*/"$model"/io_*; do
            [ -d "$task" ] || continue
            rel=${task#"$RUN_DIR"/}
            mkdir -p "$BACKUP/$(dirname "$rel")"
            if [ -e "$BACKUP/$rel" ]; then
                say "refusing to overwrite $BACKUP/$rel -- look at it and move it yourself"
                exit 1
            fi
            mv "$task" "$BACKUP/$rel" || { say "could not move $rel"; exit 1; }
            moved=$((moved + 1))
        done
    done
    say "moved $moved reasoning-on io task folder(s) to $BACKUP"

    # Drive keeps what the run folder no longer has (the upload is a copy, not a
    # mirror), so the old answers would otherwise sit beside the new ones.
    # --drive-use-trash: recoverable from Drive's trash for 30 days.
    if rclone delete "$REMOTE/datasets" --drive-use-trash --tpslimit 4 --retries 5 \
        --include "/*/gpt-5.6-luna-openrouter/io_*/**" \
        --include "/*/gemini-3.8-flash-openrouter/io_*/**" >>"$LOG" 2>&1; then
        say "moved the same folders' files on Drive to Drive's trash"
    else
        say "WARNING: could not clear them on Drive; the re-run overwrites the same files, a few old ones may remain"
    fi
    touch "$MARKER"
fi

# -- pass 1: io ---------------------------------------------------------------
say "pass 1/2: io on every dataset -- luna reasoning off, gemini low effort"
.venv/bin/abench run configs/runs/openrouter_trio_io.yaml --resume "$RUN" >>"$LOG" 2>&1
rc=$?
say "pass 1 exited with $rc"
# 0 finished; 3 finished with a failed task (still worth going on); anything
# else -- 130 stopped, 1/2 could not run -- ends here.
case $rc in 0|3) ;; *) say "not starting pass 2"; exit "$rc" ;; esac

# -- pass 2: cot --------------------------------------------------------------
say "pass 2/2: cot on every dataset, settings unchanged"
.venv/bin/abench run configs/runs/openrouter_trio.yaml --resume "$RUN" --set 'modes.prompt_modes=[cot]' >>"$LOG" 2>&1
rc=$?
say "pass 2 exited with $rc"
exit "$rc"
