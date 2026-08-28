#!/usr/bin/env bash
# Setup script for an o3b superproject checkout on the *local* machine.
#
#   bash third_party/o3b/setup/setup_local.sh
#
# All the work is done by setup/setup_slurm.sh next to this file; this script
# only resolves the parameters `o3b platform setup` would otherwise inject from
# a platform config. That indirection is the point: it runs straight after a
# `git pull`, before o3b (and therefore the `o3b` CLI) is installed at all.
#
# Differences from the cluster path, mirroring `_run_platform_setup`'s
# `ssh: False` branch in o3b/cli.py:
#   * no clone, no pull, no submodule update -- the working tree you are
#     editing is left exactly as it is (see --pull / --submodules below)
#   * the env is <repo>/venv, the layout CLAUDE.md documents, instead of the
#     tagged venv_py310_cu124_torch26 the cluster uses
#   * no credentials are staged and ~/.netrc is left alone; local git/wandb
#     already have whatever they need
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O3B_ROOT="$(dirname "${SCRIPT_DIR}")"   # …/third_party/o3b

usage() {
    cat <<EOF
usage: bash ${BASH_SOURCE[0]} [options]

  --deps a,b        optional dependency groups, as in a platform config's
                    \`deps:\` (trinity, diff3f, densematcher, genpose2,
                    morpheus, geco, mrf3dtopo, magicpony, partfield).
                    Default: none.
  --python X.Y      python for the venv (default: the first of 3.12, 3.11, 3.13,
                    3.10 that is installed *and* can build a venv; an existing
                    venv keeps whatever python it was made with)
  --torch X.Y.Z     torch version (default: ${TORCH_VERSION:-2.6.0})
  --cuda DIR        CUDA toolkit for the extension builds
                    (default: autodetected under /usr/local)
  --venv DIR        virtualenv to create/reuse (default: <repo>/venv)
  --conda           install into a conda env under --path-conda instead
  --path-conda DIR  miniconda prefix for --conda (default: \$HOME/miniconda3)
  --pull            git pull the superproject first (off: your tree is not touched)
  --submodules      git submodule update --init --recursive first
                    (limit with SKIP_SUBMODULES=<paths>)
  -h, --help        this message

Every option also has an environment-variable form -- DEPS, PYTHON_VERSION,
TORCH_VERSION, PATH_CUDA, VENV_PATH, USE_CONDA, PATH_CONDA, PULL,
PULL_SUBMODULES, SKIP_SUBMODULES, BRANCH -- and anything already exported wins
over the defaults below (but not over an explicit flag).
EOF
}

DEPS="${DEPS:-}"
PULL="${PULL:-false}"
PULL_SUBMODULES="${PULL_SUBMODULES:-false}"
USE_CONDA="${USE_CONDA:-false}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
BRANCH="${BRANCH:-main}"
SKIP_SUBMODULES="${SKIP_SUBMODULES-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --deps)        DEPS="$2"; shift 2 ;;
        --deps=*)      DEPS="${1#*=}"; shift ;;
        --python)      PYTHON_VERSION="$2"; shift 2 ;;
        --python=*)    PYTHON_VERSION="${1#*=}"; shift ;;
        --torch)       TORCH_VERSION="$2"; shift 2 ;;
        --torch=*)     TORCH_VERSION="${1#*=}"; shift ;;
        --cuda)        PATH_CUDA="$2"; shift 2 ;;
        --cuda=*)      PATH_CUDA="${1#*=}"; shift ;;
        --venv)        VENV_PATH="$2"; shift 2 ;;
        --venv=*)      VENV_PATH="${1#*=}"; shift ;;
        --conda)       USE_CONDA=true; shift ;;
        --path-conda)  PATH_CONDA="$2"; USE_CONDA=true; shift 2 ;;
        --path-conda=*) PATH_CONDA="${1#*=}"; USE_CONDA=true; shift ;;
        --pull)        PULL=true; shift ;;
        --submodules)  PULL_SUBMODULES=true; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

_is_true() { case "${1:-}" in true|True|TRUE|1|yes|Yes) return 0 ;; *) return 1 ;; esac; }

# ── repo ──────────────────────────────────────────────────────────────────────
# The superproject when o3b is checked out as a submodule (the normal case);
# otherwise whatever repo this script sits in. setup_slurm.sh rebuilds the path
# as ${PATH_WS}/${REPO_NAME}, so both halves have to come from the same root.
REPO_ROOT="${REPO_ROOT:-}"
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(git -C "${O3B_ROOT}" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
fi
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(git -C "${O3B_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [ -z "${REPO_ROOT}" ]; then
    REPO_ROOT="$(dirname "$(dirname "${O3B_ROOT}")")"   # …/<repo>/third_party/o3b → <repo>
fi
if [ ! -f "${REPO_ROOT}/third_party/o3b/pyproject.toml" ]; then
    echo "ERROR: ${REPO_ROOT} does not look like an o3b superproject" >&2
    echo "       (no third_party/o3b/pyproject.toml). Set REPO_ROOT explicitly." >&2
    exit 1
fi

# ── github token ──────────────────────────────────────────────────────────────
# `o3b platform setup` resolves credentials.github.token through hydra; this
# script runs before o3b (and omegaconf) is installed, so read the same two
# files directly. Same precedence as the hydra chain in credentials/default.yaml
# (`optional custom@: default_custom`): the custom override wins, and an
# exported GITHUB_TOKEN wins over both.
CRED_DIR="${O3B_ROOT}/src/configs/platform/credentials"
CRED_CUSTOM="${CRED_DIR}/custom/default_custom.yaml"

_yaml_github_token() {
    # Print the `token:` under a top-level `github:` key. Not a YAML parser --
    # the credentials files are two levels deep and never quote anything exotic.
    [ -f "$1" ] || return 0
    awk '
        /^[[:space:]]*#/ { next }                                  # comment line
        /^[^[:space:]#]/ { in_github = ($0 ~ /^github:[[:space:]]*$/); next }
        in_github && $1 == "token:" {
            sub(/^[[:space:]]*token:[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")                            # trailing comment
            gsub(/^["'"'"']|["'"'"']$/, "")                        # surrounding quotes
            sub(/[[:space:]]+$/, "")
            if ($0 != "") value = $0                               # last one wins, as in yaml
        }
        END { if (value != "") print value }
    ' "$1"
}

GITHUB_TOKEN="${GITHUB_TOKEN:-}"
TOKEN_SRC="${GITHUB_TOKEN:+the environment}"
if [ -z "${GITHUB_TOKEN}" ]; then
    for _f in "${CRED_DIR}/default.yaml" "${CRED_CUSTOM}"; do
        _t="$(_yaml_github_token "${_f}")"
        # "..." is the placeholder every credentials key ships with; passing it
        # on is worse than passing nothing (it shadows the credential git would
        # otherwise find and fails with a confusing auth error).
        case "${_t}" in
            ""|"..."|None) ;;
            *) GITHUB_TOKEN="${_t}"; TOKEN_SRC="${_f#"${REPO_ROOT}/"}" ;;
        esac
    done
fi

# ── python ────────────────────────────────────────────────────────────────────
# o3b needs >=3.10. An interpreter is only a candidate if it can actually build
# a venv: debian/ubuntu ship ensurepip in a separate pythonX.Y-venv package, and
# without it `python -m venv` fails *after* creating the tree, leaving a
# directory with bin/python and no bin/activate.
_python_ok() { command -v "python${1}" >/dev/null 2>&1 && "python${1}" -c "import ensurepip" >/dev/null 2>&1; }

_PY_CANDIDATES="3.12 3.11 3.13 3.10"

# An existing venv settles the question -- setup_slurm.sh reuses it, so its
# interpreter is the one that counts and no new venv has to be creatable.
_venv_ready=""
if ! _is_true "${USE_CONDA}" && [ -f "${VENV_PATH:-${REPO_ROOT}/venv}/bin/activate" ]; then
    _venv_ready="${VENV_PATH:-${REPO_ROOT}/venv}"
    PYTHON_VERSION="${PYTHON_VERSION:-$("${_venv_ready}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')}"
fi

if [ -z "${PYTHON_VERSION:-}" ]; then
    for _v in ${_PY_CANDIDATES}; do
        if _python_ok "${_v}"; then PYTHON_VERSION="${_v}"; break; fi
    done
fi
if [ -z "${PYTHON_VERSION:-}" ]; then
    echo "ERROR: found no python3.10+ that can create a virtualenv." >&2
    for _v in ${_PY_CANDIDATES}; do
        if command -v "python${_v}" >/dev/null 2>&1; then
            echo "       python${_v} is installed but has no ensurepip module:" >&2
            echo "         sudo apt install python${_v}-venv" >&2
        fi
    done
    echo "       Then re-run, or pass --python X.Y / --conda." >&2
    exit 1
fi
if [ -z "${_venv_ready}" ] && ! _is_true "${USE_CONDA}" && ! _python_ok "${PYTHON_VERSION}"; then
    # explicit --python / PYTHON_VERSION that cannot build a venv
    echo "ERROR: python${PYTHON_VERSION} cannot create a virtualenv." >&2
    if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        echo "       It is installed but has no ensurepip module:" >&2
        echo "         sudo apt install python${PYTHON_VERSION}-venv" >&2
    else
        echo "       No python${PYTHON_VERSION} on PATH." >&2
    fi
    for _v in ${_PY_CANDIDATES}; do
        if _python_ok "${_v}"; then echo "       python${_v} would work: --python ${_v}" >&2; break; fi
    done
    exit 1
fi

# A venv left behind by a failed run (tree created, ensurepip failed, so no
# bin/activate) is silently reused by every later run and never repaired.
_venv_broken="${VENV_PATH:-${REPO_ROOT}/venv}"
if ! _is_true "${USE_CONDA}" && [ -d "${_venv_broken}" ] && [ -z "${_venv_ready}" ]; then
    echo "NOTE: ${_venv_broken} exists but has no bin/activate -- an earlier venv"
    echo "      creation failed there. It holds no packages, so it is discarded"
    echo "      (venv --clear) and rebuilt with python${PYTHON_VERSION}."
    echo
fi

# ── cuda ──────────────────────────────────────────────────────────────────────
# Only needed to *build* the CUDA extensions (pointnet2, Mask2Former, sm-mrf);
# the torch wheels bring their own runtime. /usr/local/cuda is usually a symlink,
# and setup_slurm.sh derives the wheel tag from the directory name, so resolve it
# and take the version from nvcc rather than from the basename.
if [ -z "${PATH_CUDA:-}" ]; then
    for _c in /usr/local/cuda-12.4 /usr/local/cuda /opt/cuda; do
        if [ -x "${_c}/bin/nvcc" ]; then PATH_CUDA="$(readlink -f "${_c}")"; break; fi
    done
fi
PATH_CUDA="${PATH_CUDA:-/usr/local/cuda-12.4}"
if [ -z "${CUDA_VERSION:-}" ]; then
    if [ -x "${PATH_CUDA}/bin/nvcc" ]; then
        CUDA_VERSION="$("${PATH_CUDA}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    else
        CUDA_VERSION="$(basename "${PATH_CUDA}" | sed 's/^cuda-//')"
    fi
fi
if [ ! -x "${PATH_CUDA}/bin/nvcc" ] && ! _is_true "${USE_CONDA}"; then
    echo "WARNING: no nvcc at ${PATH_CUDA}/bin/nvcc -- CUDA extension builds will fail."
    echo "         Pass --cuda /usr/local/cuda-XX.Y, or --conda to get the toolkit from conda."
fi
# The pytorch/pyg indices only publish a handful of cuXXX variants; anything else
# gives a 404 halfway through the install rather than at the start.
case "${CUDA_VERSION}" in
    11.8|12.4|12.6|12.8) ;;
    *) echo "WARNING: cu$(echo "${CUDA_VERSION}" | tr -d '.') has no torch ${TORCH_VERSION} wheel index;"
       echo "         the torch install will likely 404. Pass --cuda <dir with a supported"
       echo "         version>, or set CUDA_VERSION=12.4 to build against 12.4 wheels." ;;
esac

# ── deps ──────────────────────────────────────────────────────────────────────
# One INSTALL_<DEP>=true per group, plus the DEPS_TAG that names the env --
# codepoint-sorted and "_"-joined, exactly as `_resolve_env_layout` builds it.
DEPS_TAG=""
_install_flags=()
if [ -n "${DEPS}" ]; then
    _deps_sorted="$(printf '%s\n' ${DEPS//,/ } | sed '/^[[:space:]]*$/d' | LC_ALL=C sort)"
    DEPS_TAG="$(printf '%s' "${_deps_sorted}" | paste -sd_ -)"
    for _dep in ${_deps_sorted}; do
        _install_flags+=("INSTALL_$(printf '%s' "${_dep}" | tr '[:lower:]' '[:upper:]')=true")
    done
fi

# ── env layout ────────────────────────────────────────────────────────────────
if _is_true "${USE_CONDA}"; then
    PATH_CONDA="${PATH_CONDA:-${HOME}/miniconda3}"
    VENV_PATH=""
    # left empty: setup_slurm.sh names it ${PATH_CONDA}/envs/<repo>_<env tag>
    CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
    _env_desc="conda ${PATH_CONDA}/envs/$(basename "${REPO_ROOT}")_py${PYTHON_VERSION//./}_cu${CUDA_VERSION//./}_torch$(echo "${TORCH_VERSION}" | cut -d. -f1,2 | tr -d '.')${DEPS_TAG:+_${DEPS_TAG}}"
else
    PATH_CONDA="${PATH_CONDA:-${HOME}/miniconda3}"
    VENV_PATH="${VENV_PATH:-${REPO_ROOT}/venv}"
    CONDA_ENV_PATH=""
    _env_desc="venv ${VENV_PATH}"
fi

echo "=== $(basename "${REPO_ROOT}") local setup ==="
echo "  repo   : ${REPO_ROOT}"
echo "  python : ${PYTHON_VERSION}   torch: ${TORCH_VERSION}"
echo "  cuda   : ${PATH_CUDA} (${CUDA_VERSION})"
echo "  env    : ${_env_desc}"
echo "  deps   : ${DEPS:-<none>}"
echo "  pull   : ${PULL} (submodules: ${PULL_SUBMODULES})"
echo "  github : ${TOKEN_SRC:-<no token>}"
echo

if [ -z "${GITHUB_TOKEN}" ]; then
    cat <<EOF
No github token configured. Set credentials.github.token in

    ${CRED_CUSTOM#"${REPO_ROOT}/"}

(create it from custom/default_custom_template.yaml next to it; the file is
gitignored, which is why a fresh checkout has none):

    github:
      token: "ghp_…"

Public dependencies (nvdiffrast, pytorch3d, …) install fine without it, so this
setup continues -- but anything reaching a private GenIntel repo falls back to
whatever your local git is already authenticated with, and fails if that is
nothing. That is only reached with --pull / --submodules.

EOF
fi

# ── hand over to setup_slurm.sh ───────────────────────────────────────────────
# WANDB_API_KEY stays empty on purpose -- locally `wandb login` has already
# written ~/.netrc, and setup_slurm.sh only rewrites that file when handed a key.
# GITHUB_TOKEN does get passed (resolved above), so the credential helper
# setup_slurm.sh installs in ~/.gitconfig works here too; it declines rather
# than answering with an empty password in shells that do not export the token.
# GIT_TERMINAL_PROMPT=1 undoes the fail-fast setup_slurm.sh sets for compute
# nodes, so an https fetch that finds no credentials can still ask.
exec env \
    PATH_WS="$(dirname "${REPO_ROOT}")" \
    REPO_NAME="$(basename "${REPO_ROOT}")" \
    REPO_URL="" \
    BRANCH="${BRANCH}" \
    PULL="${PULL}" \
    PULL_SUBMODULES="${PULL_SUBMODULES}" \
    SKIP_SUBMODULES="${SKIP_SUBMODULES}" \
    PULL_ONLY=false \
    CREDS_ONLY=false \
    CREDENTIALS_SRC="" \
    CREDENTIALS_DEST="" \
    GITHUB_TOKEN="${GITHUB_TOKEN:-}" \
    GIT_TERMINAL_PROMPT=1 \
    WANDB_API_KEY="" \
    MODULES="" \
    PYTHON_VERSION="${PYTHON_VERSION}" \
    TORCH_VERSION="${TORCH_VERSION}" \
    PATH_CUDA="${PATH_CUDA}" \
    CUDA_VERSION="${CUDA_VERSION}" \
    USE_CONDA="${USE_CONDA}" \
    PATH_CONDA="${PATH_CONDA}" \
    CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    VENV_PATH="${VENV_PATH}" \
    DEPS_TAG="${DEPS_TAG}" \
    LOCK_FILE="${LOCK_FILE:-${TMPDIR:-/tmp}/o3b_setup_${USER:-$(id -un)}_$(basename "${REPO_ROOT}").lock}" \
    "${_install_flags[@]}" \
    bash "${SCRIPT_DIR}/setup_slurm.sh"
