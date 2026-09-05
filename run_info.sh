#!/bin/bash
# rlmadp on the CSE infolab hosts (ant/bee/cat/dog/elk/fox.cse.iitb.ac.in).
#
#   bash run_info.sh auto                     # server + run, ONE window
#   bash run_info.sh offline                  # no GPU, no venv, ~1 second
#   bash run_info.sh serve                    # start a vLLM root server
#   bash run_info.sh run                      # RLM against that server
#   bash run_info.sh sweep                    # chunk-size sweep
#   bash run_info.sh trace                    # per-function trigger log
#   bash run_info.sh deepdive                 # what data crosses each boundary
#
# ALWAYS UNDER TMUX. These hosts have no scheduler: nothing restarts a dead run,
# and an ssh drop kills both the server and the worker.
#
#   tmux new -s rlm
#   GPU=1 bash run_info.sh serve              # window 1
#   bash run_info.sh run                      # window 2
#
# The GPUs are shared and unreserved. A card that measured idle a minute ago may
# not be now, so `serve` re-checks free memory immediately before binding and
# asks for ROOT_NEED_MIB rather than the whole card -- that is what keeps the
# host usable by everyone else on it.
set -euo pipefail

HOST="$(hostname -s)"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"
NAS="${RLMADP_NAS:-/mnt/nas/$USER}"
STORE="${RLMADP_STORE:-$NAS/rlmadp}"
VENV="${RLMADP_VENV:-$STORE/venv-$HOST}"
ARCHIVE="${RLMADP_ARCHIVE:-$STORE/results}"
RESULTS="${RESULTS:-$ARCHIVE}"
LOGS="${LOGS:-$RESULTS/logs}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIZE="${SIZE:-3000000}"       # corpus characters
CHARS="${CHARS:-40000}"       # max chars per llm_query slice
STEPS="${STEPS:-12}"
SUBCALLS="${SUBCALLS:-8}"

# A 4B model in bf16: ~8 GB of weights, plus KV for one MAXLEN sequence, plus
# ~2 GB of activations and non-torch overhead. The root never holds the document
# -- it reads REPL observations, ~6k chars a turn -- so MAXLEN can be small, and
# that is precisely why an RLM is cheap to serve.
MAXLEN="${MAXLEN:-16384}"
ROOT_NEED_MIB="${ROOT_NEED_MIB:-12000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-14000}"
# Left unclaimed so a co-tenant growing slightly does not OOM the server.
HEADROOM_MIB="${HEADROOM_MIB:-2000}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export HF_HOME="${HF_HOME:-$STORE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
# vLLM writes anonymous usage stats to ~/.config/vllm from a background thread.
# On a full $HOME that throws OSError 122 mid-startup -- non-fatal, but it is
# noise in the log and a write to a quota this project is trying to keep clear.
# There is no reason for a lab cluster run to phone home either.
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-$STORE/config}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

serve_on() {
    # One place the vllm command lives, so `serve` and `auto` cannot drift apart.
    local idx="$1" port="$2"
    local free total frac
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$idx")
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$idx")
    [ "$free" -ge "$MIN_FREE_MIB" ] || {
        echo "gpu $idx now has only ${free} MiB free (need $MIN_FREE_MIB)" >&2; return 1; }
    # --gpu-memory-utilization is a ceiling on TOTAL DEVICE memory, and vLLM
    # charges EVERY tenant's bytes against it -- ours and other users' alike.
    # So the fraction is (what the co-tenants already hold) + (what we want for
    # ourselves), over total.
    #
    # `need/total` alone is WRONG and fails in a way that looks like something
    # else: on a card holding 25 GB of co-tenants, 12000/49140 = 0.244 tells
    # vLLM to cap TOTAL device use at 12 GB, which is already exceeded. The 8 GB
    # of weights still load, then KV allocation gets nothing and the engine dies
    # with "No available memory for the cache blocks. Try increasing
    # gpu_memory_utilization" -- pointing at the fraction rather than at the
    # co-tenants it forgot to count.
    #
    # Deriving it from `free` without adding the co-tenants back subtracts them
    # a SECOND time (free is already net of them), and the KV pool absorbs the
    # shortfall. Same bug, quieter.
    frac=$(awk -v f="$free" -v t="$total" -v h="$HEADROOM_MIB" -v n="$ROOT_NEED_MIB" \
        'BEGIN{ own = f - h; if (own > n) own = n;
                u = (t - f + own) / t;
                if (u > 0.90) u = 0.90; if (u < 0.30) u = 0.30; printf "%.3f", u }')
    echo "host $HOST  gpu $idx  free ${free}/${total} MiB  port $port" >&2
    echo "co-tenants hold $((total - free)) MiB; ours <= $ROOT_NEED_MIB MiB" >&2
    echo "gpu-memory-utilization=$frac  max-model-len=$MAXLEN" >&2
    # --served-model-name pins the name in /v1/models so the readiness check
    # cannot be satisfied by a co-tenant's server holding the port.
    # --enforce-eager -O0 skips torch.compile, minutes of startup on these
    # mixed sm_86/sm_89 hosts and a past source of init failures.
    CUDA_VISIBLE_DEVICES="$idx" vllm serve "$MODEL" \
        --served-model-name "$MODEL" \
        --port "$port" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$frac" \
        --enforce-eager -O0
}

wait_ready() {
    # Poll until the server answers with OUR model name. A cold start loads ~8 GB
    # of weights, so the wait is minutes, not seconds.
    local port="$1" waited=0 limit="${READY_TIMEOUT:-900}"
    while [ "$waited" -lt "$limit" ]; do
        if curl -sf -m 5 "http://localhost:$port/v1/models" 2>/dev/null |
                grep -qF "\"$MODEL\""; then
            return 0
        fi
        sleep 5; waited=$((waited + 5))
        [ $((waited % 60)) -eq 0 ] && echo "  ... waiting for :$port (${waited}s)" >&2
    done
    return 1
}

port_free() {
    ! curl -sf -m 3 "http://localhost:$1/v1/models" >/dev/null 2>&1
}

case "${1:-}" in auto | offline | serve | run | sweep | trace | guards | deepdive) ;;
*) echo "usage: $0 {auto|offline|serve|run|sweep|trace|guards|deepdive}" >&2; exit 2 ;;
esac
cd "$REPO"

# Resolve where run output goes. On an infolab host $ARCHIVE exists and results
# land there. Anywhere else -- a laptop, a host with no /mnt/nas -- fall back to
# ./results rather than aborting: `mkdir -p` on a missing /mnt path fails, and
# under `set -e` that killed `sweep` and `run` outright instead of running them.
resolve_results() {
    if [ -d "$ARCHIVE" ] && [ -w "$ARCHIVE" ]; then
        return
    fi
    # $ARCHIVE not existing YET is the normal first-run case, not a reason to
    # fall back: if its parent store is writable, just create it. The fallback
    # below is only for a machine with no NAS at all (a laptop), where
    # `mkdir -p` on a missing /mnt path fails and, under `set -e`, killed the
    # whole run instead of running it.
    if mkdir -p "$ARCHIVE" 2>/dev/null && [ -w "$ARCHIVE" ]; then
        return
    fi
    echo "note: $ARCHIVE not available; writing results to ./results instead." >&2
    RESULTS="$REPO/results"
    LOGS="$RESULTS/logs"
}

pick_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi on $HOST" >&2; exit 1; }
    if [ -n "${GPU:-}" ]; then echo "$GPU"; return; fi
    # IDLE first, then most free memory -- not the other way round. Free memory
    # alone picks a card that has room but is already at 46% utilisation with
    # someone else's job on it; two jobs then contend for SMs and both run slow.
    # Utilisation is the signal for "is anyone computing here", free memory only
    # for "will I fit". Filter on the second, rank on the first. Chosen live,
    # because a card idle at login may be busy by the time you get here.
    local pick
    pick=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F', ' -v need="$MIN_FREE_MIB" '$2 >= need {print $3, $2, $1}' |
        sort -k1,1n -k2,2nr | head -1 | awk '{print $3}')
    if [ -z "$pick" ]; then
        echo "no GPU with >= ${MIN_FREE_MIB} MiB free on $HOST:" >&2
        nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv >&2
        exit 1
    fi
    echo "$pick"
}

activate_venv() {
    [ -f "$VENV/bin/activate" ] || {
        echo "no venv at $VENV -- run: bash setup.sh serve" >&2; exit 1; }
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
}

# =============================================================================
case "$1" in

offline)
    # Stdlib only. No venv, no GPU, no proxy, no quota. This is the branch that
    # works on a login shell the moment you land on the host.
    echo "== offline: scripted root + extractive sub (machinery only) =="
    python3 -m rlmadp.cli --size "$SIZE" --chars "$CHARS"
    ;;

trace)
    echo "== per-function trigger log =="
    python3 -m rlmadp.tracing
    ;;

deepdive)
    # What data crosses which boundary: REPL creation, what the root is told,
    # and both sides of every sub-call.
    echo "== boundary trace =="
    python3 -m rlmadp.deepdive
    ;;

guards)
    echo "== grounding + abstention guards =="
    python3 -m rlmadp.guards_demo
    ;;

serve)
    activate_venv
    mkdir -p "$LOGS"
    serve_on "$(pick_gpu)" "$PORT" 2>&1 | tee "$LOGS/vllm.$HOST.$PORT.log"
    ;;

auto)
    # Server and run in ONE window: background the server, wait for it, run the
    # client, then shut the server down. Use this unless you want the server to
    # outlive the run (several runs against one load) -- then use serve + run.
    activate_venv
    resolve_results
    mkdir -p "$LOGS" "$RESULTS"
    GPU_IDX=$(pick_gpu)
    port_free "$PORT" || { echo "port $PORT is already serving something" >&2; exit 1; }

    SRV_LOG="$LOGS/vllm.$HOST.$PORT.log"
    serve_on "$GPU_IDX" "$PORT" >"$SRV_LOG" 2>&1 &
    SRV_PID=$!

    # Two subtleties, both learned the hard way on these hosts:
    #  1. after an INT trap runs, bash RESUMES the interrupted loop unless the
    #     handler exits -- ^C looks acknowledged while the script polls forever;
    #  2. $SRV_PID is the backgrounded subshell; killing it ORPHANS the vllm
    #     child inside, which keeps the port and ~12 GB of GPU until someone
    #     kills it by hand. pkill -P takes the children too.
    # HUP as well as INT/TERM: a dropped ssh sends SIGHUP. tmux makes that
    # unlikely; this makes it survivable, not fine.
    cleanup() {
        trap - EXIT INT TERM HUP
        echo "shutting down server (pid $SRV_PID)" >&2
        pkill -TERM -P "$SRV_PID" 2>/dev/null || true
        kill "$SRV_PID" 2>/dev/null || true
        exit "${1:-0}"
    }
    trap 'cleanup 130' INT TERM HUP
    trap 'cleanup $?' EXIT

    echo "server starting on gpu $GPU_IDX, port $PORT -- log: $SRV_LOG"
    wait_ready "$PORT" || {
        echo "server never became ready; see $SRV_LOG" >&2
        tail -20 "$SRV_LOG" >&2
        exit 1
    }
    echo "server ready. running the RLM ..."

    OUT="$RESULTS/rlmadp.$HOST.$(date +%Y%m%d-%H%M%S).log"
    python3 -m rlmadp.cli --vllm \
        --base-url "http://localhost:$PORT/v1" --model "$MODEL" \
        --size "$SIZE" --chars "$CHARS" --steps "$STEPS" --sub-calls "$SUBCALLS" \
        2>&1 | tee "$OUT"
    ;;

run)
    resolve_results
    mkdir -p "$RESULTS"
    # The client is stdlib urllib, so this side needs no venv at all -- only the
    # SERVER does. Keeping them separate means a broken serve env never blocks
    # a run against someone else's already-running server.
    curl -sf -m 10 "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\"" || {
        echo "localhost:$PORT is not serving $MODEL" >&2
        echo "  start it first:  GPU=<idx> bash run_info.sh serve" >&2
        exit 1
    }
    OUT="$RESULTS/rlmadp.$HOST.$(date +%Y%m%d-%H%M%S).log"
    echo "== rlm against $MODEL on localhost:$PORT =="
    echo "   log -> $OUT"
    python3 -m rlmadp.cli --vllm \
        --base-url "http://localhost:$PORT/v1" --model "$MODEL" \
        --size "$SIZE" --chars "$CHARS" --steps "$STEPS" --sub-calls "$SUBCALLS" \
        2>&1 | tee "$OUT"
    ;;

sweep)
    # The one measurement this repo is actually for: does a bigger slice buy a
    # better answer? Small slices break the hop chain and the run abstains.
    resolve_results
    mkdir -p "$RESULTS"
    OUT="$RESULTS/sweep.$HOST.$(date +%Y%m%d-%H%M%S).log"
    MODE=(--vllm --base-url "http://localhost:$PORT/v1" --model "$MODEL")
    curl -sf -m 10 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || {
        echo "no server on :$PORT -- sweeping the OFFLINE path instead" >&2
        MODE=()
    }
    {
        printf '%-10s %-10s %-12s %-9s %s\n' chars answer coverage subcalls end
        for c in 3000 6000 12000 24000 40000 80000; do
            # ${MODE[@]+...} guards the EMPTY-array case: under `set -u`,
            # bash 3.2 treats "${MODE[@]}" on an empty array as unbound and
            # aborts the sweep on its first row.
            python3 -m rlmadp.cli ${MODE[@]+"${MODE[@]}"} --size "$SIZE" --chars "$c" 2>/dev/null |
                awk -v c="$c" '
                    /^answer /   {a = ($3 == "None") ? "none" : "found"}
                    /^end /      {e = $3}
                    /^coverage / {v = $3}
                    /^sub-calls /{s = $3}
                    END {printf "%-10s %-10s %-12s %-9s %s\n", c, a, v, s, e}'
        done
    } | tee "$OUT"
    ;;
esac
