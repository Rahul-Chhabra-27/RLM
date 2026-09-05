# Source this in ANY shell before running uv/pip/python by hand:
#
#     source env.sh
#
# setup.sh and run_info.sh export these internally, so they are safe on their
# own. A bare shell is not: `uv pip install ...` with UV_CACHE_DIR unset writes
# to ~/.cache/uv and dies with "Disk quota exceeded (os error 122)" against the
# 30 GB $HOME cap. This file is that same redirection, reusable.
#
# It also activates the venv if one exists for this host.

_RLMADP_HOST="$(hostname -s)"
export RLMADP_NAS="${RLMADP_NAS:-/mnt/nas/$USER}"
export RLMADP_STORE="${RLMADP_STORE:-$RLMADP_NAS/rlmadp}"
export RLMADP_VENV="${RLMADP_VENV:-$RLMADP_STORE/venv-$_RLMADP_HOST}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RLMADP_STORE/cache}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$RLMADP_STORE/share}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$XDG_DATA_HOME/uv/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export HF_HOME="${HF_HOME:-$RLMADP_STORE/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Respecting an existing value is right in general -- except when that value is
# under $HOME, which is the one thing this file exists to prevent. A shell that
# already exports UV_CACHE_DIR=~/.cache/uv would otherwise be "honoured" straight
# into a 30 GB quota failure. Override those, and say so.
_rlmadp_off_home() {
    _v="$1"; _fallback="$2"
    eval "_cur=\${$_v:-}"
    case "$_cur" in
    "$HOME"/* | "$HOME")
        export "$_v=$_fallback"
        echo "  note: $_v pointed at \$HOME ($_cur) -- overridden to $_fallback"
        ;;
    esac
}
_rlmadp_off_home UV_CACHE_DIR             "$XDG_CACHE_HOME/uv"
_rlmadp_off_home UV_PYTHON_INSTALL_DIR    "$XDG_DATA_HOME/uv/python"
_rlmadp_off_home PIP_CACHE_DIR            "$XDG_CACHE_HOME/pip"
_rlmadp_off_home HF_HOME                  "$RLMADP_STORE/hf_cache"
_rlmadp_off_home HF_HUB_CACHE             "$HF_HOME/hub"
_rlmadp_off_home XDG_CACHE_HOME           "$RLMADP_STORE/cache"
_rlmadp_off_home XDG_DATA_HOME            "$RLMADP_STORE/share"
_rlmadp_off_home TORCH_HOME               "$XDG_CACHE_HOME/torch"
_rlmadp_off_home VLLM_CACHE_ROOT          "$XDG_CACHE_HOME/vllm"
_rlmadp_off_home TRITON_CACHE_DIR         "$XDG_CACHE_HOME/triton"
_rlmadp_off_home TORCHINDUCTOR_CACHE_DIR  "$XDG_CACHE_HOME/inductor"
unset -f _rlmadp_off_home 2>/dev/null || true
unset _v _fallback _cur 2>/dev/null || true

if [ -f "$RLMADP_VENV/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "$RLMADP_VENV/bin/activate"
    echo "rlmadp env ready on $_RLMADP_HOST"
    echo "  venv     $RLMADP_VENV"
    echo "  uv cache $UV_CACHE_DIR"
else
    echo "rlmadp env vars set (no venv at $RLMADP_VENV yet -- run: bash setup.sh serve)"
fi
unset _RLMADP_HOST
