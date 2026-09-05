#!/bin/bash
# rlmadp environment bootstrap for the IITB CSE infolab hosts.
#
#   bash setup.sh check     # environment audit; changes nothing
#   bash setup.sh net       # activate Internet access via IITB SSO
#   bash setup.sh core      # verify the zero-dependency path works (no installs)
#   bash setup.sh serve     # build the vLLM venv on the NAS
#
# WHY THIS SCRIPT EXISTS. The core of rlmadp is stdlib-only, so `core` needs no
# venv at all. Everything below is really about the two infolab rules that a
# plain `pip install` violates:
#
#   1. $HOME is a small, nightly-backed-up NFS quota. torch + vLLM wheels unpack
#      to several GB and will fail with "OSError: [Errno 122] Disk quota
#      exceeded" -- and a large write to a stalled filer mount is uninterruptible
#      (hard,timeo=600 -> D-state; Ctrl-C does nothing). Envs and caches go on
#      /mnt/$(hostname -s)/data/$USER instead.
#   2. Different hosts carry different NVIDIA/CUDA versions, so a venv built on
#      bee is not guaranteed to work on dog. Host-local disk keeps them separate
#      by construction -- which is the same reason the guide says to keep conda
#      and pip envs there.
#
# Long jobs belong in tmux or screen: an ssh drop otherwise kills them.
set -euo pipefail

HOST="$(hostname -s)"
# Everything lives on the NAS. $HOME is capped at 30 GB and a torch+vLLM install
# plus one model blows straight through it; the NAS allocation is 100 GB.
NAS="${RLMADP_NAS:-/mnt/nas/$USER}"
STORE="${RLMADP_STORE:-$NAS/rlmadp}"
# The venv is per-HOST even on shared storage: infolab hosts carry different
# NVIDIA/CUDA versions, so one venv cannot serve bee and dog. Naming it by host
# keeps them apart while still honouring "store on the NAS".
VENV="${RLMADP_VENV:-$STORE/venv-$HOST}"
ARCHIVE="${RLMADP_ARCHIVE:-$STORE/results}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

# --- cache redirection --------------------------------------------------------
# Exported before ANY install runs. pip/uv/HF all default to $HOME, and doing
# this afterwards is useless -- the wheels are already on the quota.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORE/cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$STORE/share}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
# UV_CACHE_DIR is the one that bites. uv defaults to ~/.cache/uv and its wheel
# cache is GB-scale, so `uv venv` / `uv pip install` fills the 30 GB $HOME on
# its own -- and the failure arrives as "Disk quota exceeded" mid-install.
# UV_PYTHON_INSTALL_DIR matters for the same reason: uv will happily download a
# whole interpreter into ~/.local/share/uv/python if the host python is too old.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$XDG_DATA_HOME/uv/python}"
export HF_HOME="${HF_HOME:-$STORE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# nvidia-smi reports PCI order while CUDA defaults to FASTEST_FIRST, so without
# this the card you measured as idle is not the card CUDA hands you.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

usage() { echo "usage: bash setup.sh {check|net|core|serve}" >&2; exit 2; }
[ $# -ge 1 ] || usage

# =============================================================================
case "$1" in

check)
    echo "== rlmadp environment audit on $HOST =="
    echo
    case " $(id -Gn) " in
    *" infolab "*) ok "infolab member (groups: $(id -Gn))" ;;
    *) bad "not a member of infolab -- ask your advisor to add you" ;;
    esac
    echo
    echo "python"
    PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)
    if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        ok "python3 $PYV (>= 3.9, core path needs nothing else)"
    else
        bad "python3 $PYV is below 3.9"
        echo "        try one of the system installs, e.g. /opt/anaconda*/bin/python3"
        echo "        do NOT install a personal anaconda -- the guide forbids it"
    fi

    echo
    echo "storage"
    for pair in "$NAS:NAS allocation -- venv, caches, weights, results"; do
        dir="${pair%%:*}"; what="${pair#*:}"
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            # -c is GNU (the infolab hosts), -f is BSD (a mac running check).
            ok "$dir  -- $what   [$(df -BG "$dir" 2>/dev/null | awk 'NR==2{print $4" free"}')]"
        elif [ -d "$dir" ]; then
            warn "$dir exists but is not writable -- $what"
        else
            warn "$dir missing -- $what"
            echo "        ask your advisor for space here, or set RLMADP_NAS"
        fi
    done
    # Not a warning: `setup.sh serve` and `run_info.sh` create this themselves.
    if [ -d "$STORE" ]; then
        ok "$STORE  -- project subtree"
    else
        echo "        $STORE will be created on first \`setup.sh serve\` / run"
    fi
    # `quota -u` prints nothing on some of these hosts, so a silent result is
    # not proof of a large quota -- it usually means the cap is invisible.
    echo "  \$HOME quota (cap is 30 GB; nothing below should point here):"
    quota -u "$USER" 2>/dev/null | sed 's/^/    /' || echo "    (quota reported nothing -- assume it is small and tight)"

    # The whole point. uv's wheel cache alone is GB-scale and defaults to
    # ~/.cache/uv, so a single unset variable is what fills $HOME.
    echo "  cache redirection:"
    leaked=0
    for v in UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR HF_HOME HF_HUB_CACHE \
             XDG_CACHE_HOME XDG_DATA_HOME TORCH_HOME VLLM_CACHE_ROOT \
             TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR; do
        eval "val=\${$v:-}"
        case "$val" in
        "$HOME"/* | "$HOME") printf '    %-24s %s   <-- LEAKS TO $HOME\n' "$v" "$val"; leaked=1 ;;
        "") printf '    %-24s (unset)\n' "$v" ;;
        *) printf '    %-24s %s\n' "$v" "$val" ;;
        esac
    done
    [ "$leaked" -eq 0 ] && ok "no cache points at \$HOME" || bad "a cache points at \$HOME -- it will hit the 30 GB quota"

    echo
    echo "gpu"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu \
            --format=csv,noheader | sed 's/^/    /'
        echo "    (cards are SHARED and unreserved -- pick an idle one, leave headroom)"
    else
        warn "no nvidia-smi; this host has no GPU. The offline path still works."
    fi

    echo
    echo "network"
    if curl -sf -m 6 -o /dev/null https://pypi.org/simple/ 2>/dev/null; then
        ok "outbound Internet reachable"
    else
        warn "no outbound Internet (expected by default on these hosts)"
        echo "        run:  bash setup.sh net"
    fi

    echo
    echo "session"
    if [ -n "${TMUX:-}" ] || [ -n "${STY:-}" ]; then
        ok "inside tmux/screen -- a dropped ssh will not kill the run"
    else
        warn "not inside tmux/screen; start one before any long run:  tmux new -s rlm"
    fi
    ;;

net)
    # IITB SSO gateway. The token is read with `read -rs` and never appears in
    # argv or $HISTFILE -- the guide's proxy recipe puts the CSE LDAP password in
    # a shell variable, which leaves it in plaintext in ~/.bash_history.
    echo "IITB SSO Internet access."
    echo "Get an access token from https://sso.iitb.ac.in first."
    read -rp "  CC LDAP id: " SSO_ID
    read -rsp "  access token (not echoed): " SSO_TOK; echo
    curl --location-trusted -u "$SSO_ID:$SSO_TOK" \
        https://internet-sso.iitb.ac.in/login.php >/dev/null 2>&1 || true
    unset SSO_TOK
    if curl -sf -m 8 -o /dev/null https://pypi.org/simple/; then
        ok "Internet is up on $HOST"
        echo "  Access lapses after ~30 min idle. Re-run this if a download stalls."
    else
        bad "still no Internet -- check the id/token, or ask about the CSE proxy"
        exit 1
    fi
    ;;

core)
    # The whole point of keeping the core stdlib-only: this branch installs
    # nothing, needs no venv, no proxy and no quota.
    echo "== core path (zero dependencies) =="
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
        bad "python3 >= 3.9 required"; exit 1; }
    cd "$REPO"
    echo "-- offline demo"
    python3 -m rlmadp.cli --size 300000 2>/dev/null | tail -6
    echo "-- guards"
    python3 -m rlmadp.guards_demo 2>/dev/null | grep -E "^GUARD|turns" | sed 's/^/  /'
    echo
    ok "core path works. No venv was created and nothing was installed."
    echo "  Results belong on $ARCHIVE, not in \$HOME."
    ;;

serve)
    # Only needed to serve a root model for `run_info.sh run --vllm`.
    echo "== serve path (vLLM venv + all caches on the NAS) =="
    [ -d "$NAS" ] || { bad "$NAS does not exist -- ask your advisor for NAS space"; exit 1; }
    command -v nvidia-smi >/dev/null 2>&1 || { bad "no GPU on $HOST"; exit 1; }
    mkdir -p "$STORE" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$HF_HOME"

    echo "  venv    : $VENV"
    echo "  caches  : $XDG_CACHE_HOME"
    echo "  uv cache: $UV_CACHE_DIR"
    echo "  HF_HOME : $HF_HOME"
    df -BG "$STORE" | tail -1 | sed 's/^/  disk    : /'

    # uv when the host has it; otherwise venv, with a get-pip bootstrap because
    # some infolab Debian pythons ship without ensurepip.
    if [ ! -f "$VENV/bin/activate" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv venv "$VENV"
        else
            python3 -m venv "$VENV" || {
                python3 -m venv --without-pip "$VENV"
                curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python"
            }
        fi
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    # `uv pip` when available: same resolver semantics, far faster, and it
    # honours UV_CACHE_DIR so the wheels land on the NAS rather than in $HOME.
    if command -v uv >/dev/null 2>&1; then
        PIP="uv pip"
        echo "  installer: uv (cache $UV_CACHE_DIR)"
    else
        PIP="python -m pip"
        python -m pip install --upgrade pip wheel setuptools
        echo "  installer: pip (cache $PIP_CACHE_DIR)"
    fi

    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  nvidia driver on $HOST: $DRV"

    # rlmadp itself has no dependencies; this is just the import path.
    $PIP install -e "$REPO"

    # PIN vLLM. An unpinned resolve grabs a build whose torch targets a newer
    # CUDA than some infolab drivers accept, and it dies at init with "driver
    # too old". 0.8.5.post1 is the version the existing Qwen3 numbers on these
    # hosts were produced with, so results stay comparable.
    $PIP install "vllm==${VLLM_VERSION:-0.8.5.post1}"

    # PIN transformers LAST, so it wins over whatever vLLM pulled in.
    # transformers 5.x removed `all_special_tokens_extended`, which vLLM 0.8.5
    # still calls at tokenizer init -- the server then dies ~5 minutes into
    # startup with an error that reads like a GPU or readiness problem.
    $PIP install "transformers==${TRANSFORMERS_VERSION:-4.51.3}"

    NVDIR=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/lib64 2>/dev/null | tail -1 || true)
    if [ -n "$NVDIR" ]; then
        echo "export LD_LIBRARY_PATH=$NVDIR:\$LD_LIBRARY_PATH" >> "$VENV/bin/activate"
        ok "appended LD_LIBRARY_PATH=$NVDIR to the venv activate script"
    fi

    # --- probe the seams HERE, where the error can name the fix --------------
    python - <<'PYPROBE'
import torch, transformers
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"devices {torch.cuda.device_count()}")
print("  capabilities:", {torch.cuda.get_device_capability(i)
                          for i in range(torch.cuda.device_count())})
print(f"  transformers {transformers.__version__}")
PYPROBE

    # Pre-fetch weights with a visible progress bar. Letting the first
    # `vllm serve` pull ~8 GB inside its 15-minute readiness window fails as
    # "server never became ready", which points at the wrong thing entirely.
    echo "  pre-fetching $MODEL into $HF_HOME ..."
    hf download "$MODEL" || huggingface-cli download "$MODEL"

    # Reproduce the exact call vLLM makes at tokenizer init, so an incompatible
    # pin is caught now rather than five minutes into a serve.
    MODEL="$MODEL" python - <<'PYPROBE'
import os, sys
from transformers import AutoTokenizer
model = os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
if not hasattr(tok, "all_special_tokens_extended"):
    sys.exit(
        f"INCOMPATIBLE: {type(tok).__name__} has no `all_special_tokens_extended`, "
        "which vLLM 0.8.5 calls at startup. Pin transformers==4.51.3 "
        "(TRANSFORMERS_VERSION=... to override)."
    )
print(f"  tokenizer OK: {type(tok).__name__}, vocab {len(tok)}")
PYPROBE

    echo
    ok "serve env ready."
    echo "  Next:  bash run_info.sh auto        # server + run, one window"
    echo "  Or:    bash run_info.sh serve       # server only, to reuse across runs"
    echo "  (the run side needs no venv; only the server does)"
    ;;

*) usage ;;
esac
