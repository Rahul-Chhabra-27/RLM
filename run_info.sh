#!/bin/bash
# rlmadp on the CSE infolab hosts (ant/bee/cat/dog/elk/fox.cse.iitb.ac.in).
#
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

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export HF_HOME="${HF_HOME:-$STORE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

case "${1:-}" in offline | serve | run | sweep | trace | guards | deepdive) ;;
*) echo "usage: $0 {offline|serve|run|sweep|trace|guards|deepdive}" >&2; exit 2 ;;
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
    echo "note: $ARCHIVE not available; writing results to ./results instead." >&2
    echo "  On an infolab host, set RLMADP_ARCHIVE to keep output off the \$HOME quota." >&2
    RESULTS="$REPO/results"
    LOGS="$RESULTS/logs"
}

pick_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo "no nvidia-smi on $HOST" >&2; exit 1; }
    if [ -n "${GPU:-}" ]; then echo "$GPU"; return; fi
    # Most free memory first, then lowest utilisation. Chosen live, because a
    # card idle at login may be full by the time you get here.
    local pick
    pick=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F', ' -v need="$MIN_FREE_MIB" '$2 >= need {print $2, $3, $1}' |
        sort -k1,1nr -k2,2n | head -1 | awk '{print $3}')
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
    GPU_IDX=$(pick_gpu)
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_IDX")
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_IDX")
    echo "host $HOST  gpu $GPU_IDX  free ${FREE}/${TOTAL} MiB  model $MODEL  port $PORT"
    [ "$FREE" -ge "$MIN_FREE_MIB" ] || {
        echo "gpu $GPU_IDX now has only ${FREE} MiB free (need $MIN_FREE_MIB)" >&2; exit 1; }

    # Ask for a FRACTION sized to what this server actually needs, not for the
    # whole card. vLLM's default 0.9 would claim ~90% of a shared GPU and evict
    # nobody -- it would simply make the host unusable for your co-tenants.
    FRAC=$(awk -v need="$ROOT_NEED_MIB" -v tot="$TOTAL" 'BEGIN{
        f = need / tot; if (f > 0.9) f = 0.9; if (f < 0.15) f = 0.15; printf "%.3f", f }')
    echo "gpu-memory-utilization=$FRAC  max-model-len=$MAXLEN"
    mkdir -p "$LOGS"

    # --served-model-name pins the name in /v1/models to exactly $MODEL, so the
    # readiness grep in `run` cannot be satisfied by a co-tenant's server that
    # happens to hold the port. --enforce-eager -O0 skips torch.compile, which
    # on these mixed sm_86/sm_89 hosts costs minutes of startup and has been the
    # source of init failures that read as GPU problems.
    CUDA_VISIBLE_DEVICES="$GPU_IDX" vllm serve "$MODEL" \
        --served-model-name "$MODEL" \
        --port "$PORT" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$FRAC" \
        --enforce-eager -O0 2>&1 | tee "$LOGS/vllm.$HOST.$PORT.log"
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
