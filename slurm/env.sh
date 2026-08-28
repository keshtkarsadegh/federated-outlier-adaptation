#!/bin/bash
# Shared environment setup for every batch job of this project.
#
# Source this at the top of a job script:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# SITE-NEUTRAL BY CONSTRUCTION.  No account, partition, proxy or absolute path
# is baked in.  Anything site-specific is unset by default and is applied only
# when you export it, so an unconfigured site gets plain behaviour rather than
# somebody else's cluster.
#
# REQUIRED
#   FOA_PROJECT_DIR   Writable root holding data/, results/ and envs/.
#                     No default - see the refusal below for why.
#
# OPTIONAL (defaults in brackets)
#   FOA_REPO          Repository checkout          [the parent of this file]
#   FOA_ENV           Conda/venv prefix to activate  [none; see ACTIVATION]
#   FOA_RESULTS_DIR   Results root                 [$FOA_PROJECT_DIR/results]
#   FOA_DATA_DIR      Data root                    [$FOA_PROJECT_DIR/data]
#   FOA_NIST28_DIR    Packed 28x28 cache           [$FOA_DATA_DIR/nist28]
#   FOA_CACHE_DIR     Raw NIST cache               [$FOA_DATA_DIR/nist/cache]
#   FOA_ACCOUNT       Slurm account                [unset]
#   FOA_GPU_PARTITION GPU partition                [unset]
#   FOA_CPU_PARTITION CPU partition                [unset]
#   FOA_HTTP_PROXY    Outbound proxy, for sites whose compute nodes have no
#                     direct route to the internet  [unset]
#   FOA_MODULES       Space-separated environment modules to `module load`
#                     before activating, e.g. "miniforge3"  [unset]
#
# ACTIVATION.  If FOA_ENV names a conda prefix it is activated; otherwise the
# interpreter already on PATH is used, which is what a plain virtualenv or a
# `pip install -e .` checkout wants.  Either way the source tree is put on
# PYTHONPATH, so a shared environment is never mutated by a job.

set -euo pipefail

if [ -z "${FOA_PROJECT_DIR:-}" ]; then
    echo "FOA_PROJECT_DIR is not set." >&2
    echo "It is the writable root holding data/, results/ and envs/. There is" >&2
    echo "no default: one would point at a directory nobody created and turn a" >&2
    echo "missing setup into an empty study." >&2
    echo "  export FOA_PROJECT_DIR=/path/to/your/workspace" >&2
    return 78 2>/dev/null || exit 78
fi
export FOA_PROJECT_DIR

_foa_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FOA_REPO="${FOA_REPO:-$(cd "$_foa_here/.." && pwd)}"
unset _foa_here

export FOA_RESULTS_DIR="${FOA_RESULTS_DIR:-$FOA_PROJECT_DIR/results}"
export FOA_DATA_DIR="${FOA_DATA_DIR:-$FOA_PROJECT_DIR/data}"
export FOA_NIST28_DIR="${FOA_NIST28_DIR:-$FOA_DATA_DIR/nist28}"
export FOA_CACHE_DIR="${FOA_CACHE_DIR:-$FOA_DATA_DIR/nist/cache}"

# Compute nodes are headless; the plotting layer must not look for a display.
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Sites whose compute nodes have no direct route to the internet set this; the
# rest are left alone rather than pointed at a proxy that does not exist.
if [ -n "${FOA_HTTP_PROXY:-}" ]; then
    export http_proxy="$FOA_HTTP_PROXY"  https_proxy="$FOA_HTTP_PROXY"
    export HTTP_PROXY="$FOA_HTTP_PROXY"  HTTPS_PROXY="$FOA_HTTP_PROXY"
    export no_proxy="${no_proxy:-localhost,127.0.0.1}"
fi

# Environment modules, when the site uses them.
if [ -n "${FOA_MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    module load $FOA_MODULES
fi

# Conda only when asked for; otherwise whatever python is on PATH.
if [ -n "${FOA_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$FOA_ENV"
    elif [ -f "$FOA_ENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$FOA_ENV/bin/activate"
    else
        echo "FOA_ENV=$FOA_ENV names neither a conda prefix nor a venv." >&2
        return 78 2>/dev/null || exit 78
    fi
fi

cd "$FOA_REPO"
# The checkout is importable either way: an editable install, or the source
# tree on PYTHONPATH.  Exporting the source root first keeps a shared
# environment untouched when the package is not installed into it.
export PYTHONPATH="$FOA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import federated_outlier_adaptation" >/dev/null 2>&1 \
    || pip install -e . --no-deps -q

echo "host=$(hostname) repo=$FOA_REPO results=$FOA_RESULTS_DIR data=$FOA_DATA_DIR"
