#!/bin/bash
# rlmadp environment bootstrap for the IITB CSE infolab hosts.
#
#   bash setup.sh check     # environment audit; changes nothing
#   bash setup.sh net       # activate Internet access via IITB SSO
#   bash setup.sh core      # verify the zero-dependency path works (no installs)
#   bash setup.sh serve     # build the vLLM venv on host-local disk
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
LOCAL="${RLMADP_LOCAL:-/mnt/$HOST/data/$USER}"     # host-local scratch: envs, caches
ARCHIVE="${RLMADP_ARCHIVE:-/mnt/nas/$USER}"        # backed up ~monthly: results
VENV="${RLMADP_VENV:-$LOCAL/rlmadp-venv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; }

# --- cache redirection --------------------------------------------------------
# Exported before ANY install runs. pip/uv/HF all default to $HOME, and doing
# this afterwards is useless -- the wheels are already on the quota.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LOCAL/cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export HF_HOME="${HF_HOME:-$LOCAL/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
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
    echo "identity"
    # The guide is explicit: assume primary group infolab BEFORE writing to the
    # shared trees, or files land with the wrong group and collaborators cannot
    # read them. newgrp replaces the shell, so a script cannot do it for you.
    if [ "$(id -gn)" = "infolab" ]; then
        ok "primary group is infolab"
    else
        warn "primary group is '$(id -gn)', not infolab"
        echo "        run:  exec newgrp infolab      (then re-run this script)"
    fi

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
    for pair in "$LOCAL:host-local scratch (envs, caches)" \
                "$ARCHIVE:archival, backed up ~monthly (results)"; do
        dir="${pair%%:*}"; what="${pair#*:}"
        if [ -d "$dir" ] && [ -w "$dir" ]; then
            ok "$dir  -- $what   [$(df -BG "$dir" 2>/dev/null | awk 'NR==2{print $4" free"}')]"
        elif [ -d "$dir" ]; then
            warn "$dir exists but is not writable -- $what"
        else
            warn "$dir missing -- $what"
            echo "        ask your advisor for space here, or set RLMADP_LOCAL / RLMADP_ARCHIVE"
        fi
    done
    # `quota -u` prints nothing on some of these hosts, so a silent result is
    # not proof of a large quota -- it usually means the cap is invisible.
    echo "  \$HOME quota:"
    quota -u "$USER" 2>/dev/null | sed 's/^/    /' || echo "    (quota reported nothing -- assume it is small and tight)"

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
    echo "== serve path (vLLM venv on host-local disk) =="
    [ -d "$LOCAL" ] || { bad "$LOCAL does not exist -- ask your advisor for local space"; exit 1; }
    command -v nvidia-smi >/dev/null 2>&1 || { bad "no GPU on $HOST"; exit 1; }
    mkdir -p "$LOCAL" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$HF_HOME"

    echo "  venv    : $VENV"
    echo "  caches  : $XDG_CACHE_HOME"
    echo "  HF_HOME : $HF_HOME"
    df -BG "$LOCAL" | tail -1 | sed 's/^/  disk    : /'

    if [ ! -f "$VENV/bin/activate" ]; then
        python3 -m venv "$VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    python -m pip install --upgrade pip wheel setuptools

    # Driver first: vLLM/torch wheels are built against a CUDA version, and the
    # host driver is the constraint that actually decides which wheel works.
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  nvidia driver on $HOST: $DRV"
    echo "  installing rlmadp[serve] -- if this resolves a vLLM that the driver"
    echo "  rejects at runtime, pin an older one and re-run."
    pip install -e "$REPO[serve]"

    # The guide's CUDA note: point the loader at the host's hpc_sdk libraries.
    NVDIR=$(ls -d /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/lib64 2>/dev/null | tail -1 || true)
    if [ -n "$NVDIR" ]; then
        echo "export LD_LIBRARY_PATH=$NVDIR:\$LD_LIBRARY_PATH" >> "$VENV/bin/activate"
        ok "appended LD_LIBRARY_PATH=$NVDIR to the venv activate script"
    fi

    echo
    ok "serve env ready. Activate with:  source $VENV/bin/activate"
    echo "  Then:  bash run_info.sh serve      (in its own tmux window)"
    ;;

*) usage ;;
esac
