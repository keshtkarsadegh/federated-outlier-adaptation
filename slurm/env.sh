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
#   FOA_ENV           Conda/venv prefix to activate  [auto; see ACTIVATION]
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
# ACTIVATION.  FOA_ENV names a conda prefix or a virtualenv.  When it is unset
# this looks for one under $FOA_PROJECT_DIR/envs/ - `foa` first, then `fal`,
# then a single unambiguous entry - and activates what it finds.  That is a
# convention relative to a directory the caller already supplied, not a site
# assumption: a workspace with no envs/ simply uses the interpreter on PATH,
# which is what a plain `pip install -e .` checkout wants.
#
# WHY THE VERSION IS CHECKED.  Batch nodes routinely have an ancient /usr/bin
# python3 ahead of everything on PATH.  Running under it does not fail where the
# mistake was made: the job starts, the guard passes, and it dies hundreds of
# lines later inside an import with `SyntaxError: future feature annotations is
# not defined` - which reads as a broken source file and is not one.  So the
# interpreter is checked here, by name and version, and the job refuses rather
# than proceeding on a Python that cannot parse the package.

set -euo pipefail

if [ -z "${FOA_PROJECT_DIR:-}" ]; then
    echo "FOA_PROJECT_DIR is not set." >&2
    echo "It is the writable root holding data/, results/ and envs/. There is" >&2
    echo "no default: one would point at a directory nobody created and turn a" >&2
    echo "missing setup into an empty study." >&2
    echo "  export FOA_PROJECT_DIR=/path/to/your/workspace" >&2
    echo "If this is a Slurm job, the submitting shell's environment was NOT" >&2
    echo "propagated: many sites default to --export=NONE, and --export=ALL is" >&2
    echo "the default being overridden. Name the variables by value:" >&2
    echo "  sbatch --export=ALL,FOA_PROJECT_DIR=\$FOA_PROJECT_DIR,FOA_STUDY_DIR=\$FOA_STUDY_DIR ..." >&2
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

# An environment under the workspace, when the caller did not name one.
if [ -z "${FOA_ENV:-}" ]; then
    for _foa_candidate in "$FOA_PROJECT_DIR/envs/foa" "$FOA_PROJECT_DIR/envs/fal"; do
        if [ -x "$_foa_candidate/bin/python" ]; then
            FOA_ENV="$_foa_candidate"
            break
        fi
    done
    if [ -z "${FOA_ENV:-}" ] && [ -d "$FOA_PROJECT_DIR/envs" ]; then
        # Exactly one environment is unambiguous; several are not, and guessing
        # between them is how a run ends up in the wrong one.
        _foa_only=""
        for _foa_candidate in "$FOA_PROJECT_DIR"/envs/*/; do
            [ -x "${_foa_candidate}bin/python" ] || continue
            if [ -n "$_foa_only" ]; then _foa_only=""; break; fi
            _foa_only="${_foa_candidate%/}"
        done
        [ -n "$_foa_only" ] && FOA_ENV="$_foa_only"
    fi
    unset _foa_candidate _foa_only
    [ -n "${FOA_ENV:-}" ] && export FOA_ENV
fi

# Three ways in, in order of how much they set up:
#
#   conda activate   full environment, when conda is reachable
#   bin/activate     a virtualenv
#   bin/python       a conda prefix with no conda on PATH - which is the normal
#                    case in a batch job, where the shell is not interactive and
#                    the module that provides conda has not been loaded. Putting
#                    the prefix's bin/ first is enough to run it, and refusing
#                    here instead was what left a whole chain on the system
#                    python.
if [ -n "${FOA_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$FOA_ENV"
    elif [ -f "$FOA_ENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$FOA_ENV/bin/activate"
    elif [ -x "$FOA_ENV/bin/python" ]; then
        export PATH="$FOA_ENV/bin:$PATH"
    else
        echo "REFUSED: FOA_ENV=$FOA_ENV holds no bin/python." >&2
        return 78 2>/dev/null || exit 78
    fi
fi

cd "$FOA_REPO"
# The checkout is importable either way: an editable install, or the source
# tree on PYTHONPATH.  Exporting the source root first keeps a shared
# environment untouched when the package is not installed into it.
export PYTHONPATH="$FOA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# One name for the interpreter, resolved once, so every later step and every
# task line runs under the same one.
if [ -n "${FOA_ENV:-}" ] && [ -x "$FOA_ENV/bin/python" ]; then
    FOA_PYTHON="$FOA_ENV/bin/python"
else
    FOA_PYTHON="$(command -v python || command -v python3)"
fi
export FOA_PYTHON

if ! "$FOA_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "REFUSED: $FOA_PYTHON is $("$FOA_PYTHON" -V 2>&1), and this package needs 3.9+." >&2
    echo "A batch node's /usr/bin/python3 is often older than the environment you" >&2
    echo "meant to use, and running under it fails deep inside an import with a" >&2
    echo "SyntaxError that looks like a broken source file. Name the environment:" >&2
    echo "  export FOA_ENV=/path/to/env      # conda prefix or virtualenv" >&2
    return 78 2>/dev/null || exit 78
fi

if ! "$FOA_PYTHON" -c "import federated_outlier_adaptation" >/dev/null 2>&1; then
    "$FOA_PYTHON" -m pip install -e . --no-deps -q || true
    if ! "$FOA_PYTHON" -c "import federated_outlier_adaptation" >/dev/null 2>&1; then
        echo "REFUSED: $FOA_PYTHON cannot import federated_outlier_adaptation from" >&2
        echo "$FOA_REPO/src. The real error follows:" >&2
        "$FOA_PYTHON" -c "import federated_outlier_adaptation" >&2 || true
        return 78 2>/dev/null || exit 78
    fi
fi

echo "host=$(hostname) repo=$FOA_REPO results=$FOA_RESULTS_DIR data=$FOA_DATA_DIR"
echo "python=$FOA_PYTHON ($("$FOA_PYTHON" -V 2>&1))${FOA_ENV:+ env=$FOA_ENV}"
