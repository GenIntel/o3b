#!/usr/bin/env bash
# Setup script for the housecorr3d repository on a SLURM cluster.
# Invoked by `o3b platform setup -p slurm`; environment variables are injected
# by that command from the resolved platform config.
set -euo pipefail

PATH_WS="${PATH_WS:-/work/dlclarge1/sommerl-od3d}"
PATH_CUDA="${PATH_CUDA:-/usr/local/cuda-12.4}"
REPO_URL="${REPO_URL:-}"                                    # housecorr3d SSH URL
REPO_NAME="${REPO_NAME:-$(basename "${REPO_URL}" .git)}"   # e.g. HouseCorr3Dv2
BRANCH="${BRANCH:-main}"
PULL="${PULL:-true}"
PULL_SUBMODULES="${PULL_SUBMODULES:-true}"
SKIP_SUBMODULES="${SKIP_SUBMODULES:-}"
# Stop after the clone/pull/submodule/credentials phase, before touching the
# env. Where compute nodes have no route to github (setup_on_login platforms)
# the checkout only advances during setup, so `o3b bench rrun --pull` uses this
# to refresh it in seconds rather than re-running the whole dependency install.
PULL_ONLY="${PULL_ONLY:-false}"
# Stop after the same point, but without pulling: just re-install the staged
# credentials (and the wandb ~/.netrc entry below) onto an existing checkout.
CREDS_ONLY="${CREDS_ONLY:-false}"
# Written to ~/.netrc so the wandb CLI/library authenticate without an env var.
# Empty (the usual case) leaves any existing ~/.netrc completely untouched.
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_HOST="${WANDB_HOST:-api.wandb.ai}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-}"   # must be exported: the git credential helper reads it
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
INSTALL_DIFF3F="${INSTALL_DIFF3F:-false}"
INSTALL_DENSEMATCHER="${INSTALL_DENSEMATCHER:-false}"
INSTALL_GENPOSE2="${INSTALL_GENPOSE2:-false}"
INSTALL_GECO="${INSTALL_GECO:-false}"
INSTALL_MRF3DTOPO="${INSTALL_MRF3DTOPO:-false}"
INSTALL_MORPHEUS="${INSTALL_MORPHEUS:-false}"
INSTALL_MAGICPONY="${INSTALL_MAGICPONY:-false}"
INSTALL_PARTFIELD="${INSTALL_PARTFIELD:-false}"
DEPS_TAG="${DEPS_TAG:-}"   # e.g. "densematcher" or "densematcher_diff3f"; appended to venv name
GIT_RETRIES="${GIT_RETRIES:-4}"        # attempts per network-facing git step
GIT_RETRY_WAIT="${GIT_RETRY_WAIT:-20}" # seconds between those attempts

# `o3b platform setup` scp's the gitignored credentials/custom/*.yaml here before
# submitting; they are copied into the freshly cloned/pulled repo further below.
CREDENTIALS_SRC="${CREDENTIALS_SRC:-}"    # absolute path on this host
CREDENTIALS_DEST="${CREDENTIALS_DEST:-}"  # repo-relative destination directory

# When USE_CONDA is set, everything is installed into a conda env instead of a
# venv, and the CUDA toolchain comes from that env (conda -c nvidia) instead of
# the system install at PATH_CUDA. See setup/setup_conda.sh for the manual
# equivalent.
USE_CONDA="${USE_CONDA:-false}"
PATH_CONDA="${PATH_CONDA:-${PATH_WS}/miniconda3}"
CUDA_VERSION="${CUDA_VERSION:-$(basename "${PATH_CUDA}" | sed 's/cuda-//')}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"

_is_true() { case "${1:-}" in true|True|TRUE|1|yes|Yes) return 0 ;; *) return 1 ;; esac; }

# ── CUDA architectures for extension builds ───────────────────────────────────
# Set from the platform config (`torch_cuda_arch_list`). Leaving it unset makes
# torch auto-detect from the visible cards, and on Grace-Hopper that path dies:
# torch.cuda.get_arch_list() reports sm_90a and _get_cuda_arch_flags does
# int(arch.split('_')[1]) -> int('90a') -> ValueError. Setting it skips that
# detection entirely (and makes builds independent of the builder's GPU).
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    export TORCH_CUDA_ARCH_LIST
    echo "--- TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ---"
fi

# pytorch3d, installed identically by the diff3f and morpheus blocks below.
# Prefers the pre-built wheel matching the torch+cuda combo, e.g.
# TORCH_VERSION=2.6.0 + CUDA_TAG=cu124 -> pytorch3d==0.7.8+pt2.6.0cu124.
# That index publishes win_amd64/linux_x86_64/macosx only, so on aarch64
# (JUPITER) nothing matches and we fall back to building from source.
_install_pytorch3d() {
    local tag="pt${TORCH_VERSION}${CUDA_TAG}"
    echo "--- pip install pytorch3d==0.7.8+${tag} ---"
    if pip install "pytorch3d==0.7.8+${tag}" \
        --extra-index-url https://miropsota.github.io/torch_packages_builder; then
        return 0
    fi
    echo "--- no pre-built pytorch3d for $(uname -m); building v0.7.8 from source ---"
    pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
}

# As for nvdiffrast: a pytorch3d left over from a different torch version is
# "already satisfied" to pip but fails to import against the current libtorch.
_install_pytorch3d_checked() {
    _install_pytorch3d
    if python -c "import pytorch3d, pytorch3d.renderer" >/dev/null 2>&1; then
        return 0
    fi
    echo "--- pytorch3d does not import against torch ${TORCH_VERSION}; rebuilding from source ---"
    pip install --no-build-isolation --force-reinstall --no-cache-dir \
        "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
    python -c "import pytorch3d, pytorch3d.renderer" >/dev/null 2>&1 \
        || echo "WARNING: pytorch3d still does not import -- morpheus/diff3f will fall back to GT/oracle."
}

# ── modules ───────────────────────────────────────────────────────────────────
# JSC systems (juwels/jupiter) ship no usable system python, git or CUDA: those
# come from Lmod. MODULES is a space-separated list from the platform config's
# `modules:`; empty on clusters that need none (the LMB cluster).
MODULES="${MODULES:-}"
if [ -n "${MODULES}" ]; then
    echo "--- module load ${MODULES} ---"
    # Lmod's shell functions are not `set -u`/`set -e` clean
    set +eu
    # non-interactive shells often lack the Lmod init that ~/.bashrc performs
    if ! command -v module >/dev/null 2>&1 && [ -f /usr/share/lmod/lmod/init/bash ]; then
        . /usr/share/lmod/lmod/init/bash
    fi
    for _m in ${MODULES}; do
        module load "${_m}" || echo "WARNING: module load ${_m} failed"
    done
    set -eu
    module list 2>&1 || true
fi

# Morpheus needs pytorch3d (same wheel as diff3f) plus the complete GenPose2
# setup; force the GenPose2 block on so its deps are not duplicated here.
if [ "${INSTALL_MORPHEUS}" = "true" ] || [ "${INSTALL_MORPHEUS}" = "True" ]; then
    INSTALL_GENPOSE2="true"
fi

# CUDA lays its per-target tree out under targets/<triple>/. The triple follows
# the CPU, not the GPU: sbsa-linux on arm64 (JUPITER's Grace-Hopper nodes),
# x86_64-linux everywhere else. Hardcoding x86_64 silently yields a CPATH that
# does not exist, and the extension builds then fail on a missing cuda_runtime.h.
case "$(uname -m)" in
    aarch64|arm64) CUDA_TARGET_TRIPLE="sbsa-linux" ;;
    *)             CUDA_TARGET_TRIPLE="x86_64-linux" ;;
esac

# Point the CUDA toolchain env at $1 (a system CUDA dir, or $CONDA_PREFIX when
# the toolkit is installed into the conda env).
_export_cuda_env() {
    export CUDA_HOME="$1"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"                                    # nvcc requires this (pointnet install)
    export CPATH="${CPATH:-}:${CUDA_HOME}/targets/${CUDA_TARGET_TRIPLE}/include"  # pycuda requires this
    export LIBRARY_PATH="${LIBRARY_PATH:-}:${CUDA_HOME}/targets/${CUDA_TARGET_TRIPLE}/lib"
}

# Under conda the toolchain is exported after the env is activated (CONDA_PREFIX
# is only known then); exporting the system CUDA here would shadow it on PATH.
if ! _is_true "${USE_CONDA}"; then
    _export_cuda_env "${PATH_CUDA}"
fi

REPO_PATH="${PATH_WS}/${REPO_NAME}"

# ── github auth ───────────────────────────────────────────────────────────────
# The token is never persisted anywhere: the credential helper written to
# ~/.gitconfig references ${GITHUB_TOKEN} literally, so it always resolves to
# whatever token the current job was started with. Rotating the token needs no
# cleanup on the cluster.
_git_setup_credentials() {
    # Purge legacy `url."https://<token>@github.com/".insteadOf` rewrites. The
    # token was part of the config *key*, so every rotation appended another
    # section instead of replacing one -- and git breaks insteadOf ties by
    # config order, i.e. the oldest (dead) token kept winning.
    local _k
    for _k in $(git config --global --name-only --get-regexp '^url\..*github\.com.*\.insteadof$' 2>/dev/null || true); do
        echo "--- dropping stale git rewrite: ${_k%.insteadof}"
        git config --global --remove-section "${_k%.insteadof}" 2>/dev/null || true
    done
    if [ -n "${GITHUB_TOKEN}" ]; then
        git config --global --replace-all credential."https://github.com".helper \
            '!f() { [ "$1" = get ] || exit 0; echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f'
    else
        echo "WARNING: GITHUB_TOKEN is empty -- private github fetches will fail"
    fi
    # A stale entry in the `store` helper's file would still be offered first.
    if [ -f "${HOME}/.git-credentials" ] && grep -q 'github\.com' "${HOME}/.git-credentials" 2>/dev/null; then
        echo "WARNING: ${HOME}/.git-credentials holds a github.com entry that may shadow GITHUB_TOKEN"
    fi
    export GIT_TERMINAL_PROMPT=0   # fail fast instead of hanging on a compute node
}

# github is only reachable through tfproxy, which intermittently answers CONNECT
# with 503 ("CONNECT tunnel failed"). Under `set -e` a single hiccup aborts the
# whole setup job seconds after it starts, so retry the network-facing git steps.
_git_retry() {
    local _n=1
    until "$@"; do
        if [ "${_n}" -ge "${GIT_RETRIES}" ]; then
            echo "ERROR: '$*' failed after ${_n} attempts" >&2
            return 1
        fi
        echo "--- git step failed, retry ${_n}/${GIT_RETRIES} in ${GIT_RETRY_WAIT}s: $* ---"
        sleep "${GIT_RETRY_WAIT}"
        _n=$((_n + 1))
    done
}

# Strip an embedded token from a repo's origin URL so the credential helper is
# consulted (git ignores helpers when the URL already carries a password).
_git_scrub_token_urls() {
    local repo="$1" url
    url="$(git -C "${repo}" remote get-url origin 2>/dev/null || true)"
    case "${url}" in
        https://*@github.com/*)
            echo "--- scrubbing embedded token from origin URL"
            git -C "${repo}" remote set-url origin "https://github.com/${url#*@github.com/}"
            ;;
    esac
}

_git_setup_credentials

# e.g. /usr/local/cuda-12.4 + 3.10 + 2.6.0  →  venv_py310_cu124_torch26
# Under conda the CUDA version comes from CUDA_VERSION (the conda toolkit),
# not from the basename of the system PATH_CUDA.
# Always from CUDA_VERSION (o3b defaults it to the basename of PATH_CUDA, so the
# LMB tag stays cu124). The basename alone breaks on JSC, whose install is
# .../CUDA/12 with nvcc 12.6 -- that gave "cu12" and a 404 pytorch index URL.
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
PY_TAG="py$(echo "${PYTHON_VERSION}" | sed 's/\.//')"
TORCH_TAG="torch$(echo "${TORCH_VERSION}" | cut -d. -f1,2 | sed 's/\.//')"
ENV_TAG="${PY_TAG}_${CUDA_TAG}_${TORCH_TAG}${DEPS_TAG:+_${DEPS_TAG}}"
# VENV_PATH / CONDA_ENV_PATH may be overridden in the environment (o3b passes
# them explicitly so it can activate the same env for `o3b run` / `o3b runi`;
# the local setup reuses an existing ./venv).
VENV_PATH="${VENV_PATH:-${REPO_PATH}/venv_${ENV_TAG}}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-${PATH_CONDA}/envs/${REPO_NAME}_${ENV_TAG}}"

LOCK_FILE="${PATH_WS}/setup_slurm.lock"
exec 200>"${LOCK_FILE}"
echo "--- acquiring setup lock (${LOCK_FILE}) ---"
flock -x 200   # blocks until no other setup_slurm.sh holds the lock

echo "=== housecorr3d slurm setup ==="
echo "  repo  : ${REPO_PATH}"
echo "  branch: ${BRANCH}"
if _is_true "${USE_CONDA}"; then
    echo "  cuda  : conda cuda-toolkit ${CUDA_VERSION} (${PATH_CONDA})"
    echo "  env   : conda ${CONDA_ENV_PATH}"
else
    echo "  cuda  : ${CUDA_HOME}"
    echo "  venv  : ${VENV_PATH}"
fi

if [ ! -d "${REPO_PATH}" ]; then
    if [ -z "${REPO_URL}" ]; then
        echo "ERROR: ${REPO_PATH} does not exist and REPO_URL is not set."
        exit 1
    fi
    echo "--- git clone ${REPO_URL} ---"
    _git_retry git clone "${REPO_URL}" "${REPO_PATH}"
fi

cd "${REPO_PATH}"

# repos cloned by older revisions have the token baked into .git/config
_git_scrub_token_urls "${REPO_PATH}"

if [ "${PULL}" = "true" ] || [ "${PULL}" = "True" ]; then
    echo "--- git pull origin ${BRANCH} ---"
    _git_retry git pull origin "${BRANCH}"
fi

if [ "${PULL_SUBMODULES}" = "true" ] || [ "${PULL_SUBMODULES}" = "True" ]; then
    echo "--- git submodule update ---"
    # resets submodule URLs to the token-free ones in .gitmodules
    git submodule sync --recursive
    if [ -z "${SKIP_SUBMODULES}" ]; then
        _git_retry git submodule update --init --recursive
    else
        git submodule init
        for sub in $(git submodule status | awk '{print $2}'); do
            _skip=false
            for s in ${SKIP_SUBMODULES}; do
                [ "$sub" = "$s" ] && _skip=true && break
            done
            if [ "$_skip" = "false" ]; then
                _git_retry git submodule update --init --recursive -- "$sub"
            fi
        done
    fi
fi

# ── credentials ───────────────────────────────────────────────────────────────
# credentials/custom/*.yaml is gitignored, so neither the clone nor the pull ever
# brings it along and the config falls back to the "..." placeholders in
# credentials/default.yaml. `o3b platform setup` stages the local copies in
# CREDENTIALS_SRC; install them now that the submodule holding the destination
# has been checked out.
if [ -n "${CREDENTIALS_SRC}" ] && [ -n "${CREDENTIALS_DEST}" ] && [ -d "${CREDENTIALS_SRC}" ]; then
    echo "--- installing credentials into ${CREDENTIALS_DEST} ---"
    mkdir -p "${REPO_PATH}/${CREDENTIALS_DEST}"
    # per file, so the tracked *_template.yaml sitting in the same directory
    # keeps its 644 and does not show up as a git mode change
    for _cred in "${CREDENTIALS_SRC}"/*.yaml; do
        [ -f "${_cred}" ] || continue
        echo "      $(basename "${_cred}")"
        install -m 600 "${_cred}" "${REPO_PATH}/${CREDENTIALS_DEST}/$(basename "${_cred}")"
    done
fi

# ── wandb login ───────────────────────────────────────────────────────────────
# The yaml above is only read by o3b itself; the `wandb` CLI and library
# authenticate from ~/.netrc. Write the entry here so `o3b bench wbsync` (and any
# online run) works on this host without exporting anything. Done in shell rather
# than via `wandb login` because this runs before the venv exists.
if [ -n "${WANDB_API_KEY}" ]; then
    echo "--- wandb login (${WANDB_HOST}) ---"
    _netrc="${HOME}/.netrc"
    _netrc_tmp="$(mktemp "${HOME}/.netrc.XXXXXX")"
    chmod 600 "${_netrc_tmp}"
    # drop any existing block for this machine, keeping every other entry: the
    # block runs from its `machine <host>` line to the next machine/default line
    if [ -f "${_netrc}" ]; then
        awk -v h="${WANDB_HOST}" '
            $1 == "machine" { skip = ($2 == h) }
            $1 == "default" { skip = 0 }
            !skip { print }
        ' "${_netrc}" > "${_netrc_tmp}"
    fi
    {
        echo "machine ${WANDB_HOST}"
        echo "  login user"
        echo "  password ${WANDB_API_KEY}"
    } >> "${_netrc_tmp}"
    mv -f "${_netrc_tmp}" "${_netrc}"
    chmod 600 "${_netrc}"
    echo "      wrote ${_netrc} (machine ${WANDB_HOST})"
fi

if _is_true "${PULL_ONLY}" || _is_true "${CREDS_ONLY}"; then
    if _is_true "${CREDS_ONLY}"; then
        echo "--- CREDS_ONLY: credentials installed, skipping env/dependency install ---"
    else
        echo "--- PULL_ONLY: checkout refreshed, skipping env/dependency install ---"
    fi
    echo "    $(git -C "${REPO_PATH}" log --oneline -1)"
    git -C "${REPO_PATH}" submodule --quiet foreach \
        'echo "    $displaypath $(git log --oneline -1)"' || true
    exit 0
fi

if _is_true "${USE_CONDA}"; then
    if [ ! -x "${PATH_CONDA}/bin/conda" ]; then
        echo "ERROR: no conda at ${PATH_CONDA}/bin/conda -- install miniconda there first:" >&2
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" >&2
        echo "  bash Miniconda3-latest-Linux-x86_64.sh -b -p ${PATH_CONDA}" >&2
        exit 1
    fi
    # conda's shell hook and activate scripts are not `set -u` clean
    case $- in *u*) _had_u=1 ;; *) _had_u=0 ;; esac
    set +u
    eval "$("${PATH_CONDA}/bin/conda" shell.bash hook)"
    if [ ! -d "${CONDA_ENV_PATH}" ]; then
        echo "--- conda create -p ${CONDA_ENV_PATH} python=${PYTHON_VERSION} ---"
        conda create -p "${CONDA_ENV_PATH}" "python=${PYTHON_VERSION}" -y
    fi
    echo "--- activate conda env ${CONDA_ENV_PATH} ---"
    conda activate "${CONDA_ENV_PATH}"
    if [ "${_had_u}" = "1" ]; then set -u; fi
    hash -r   # clear cached PATH lookups (e.g. a ~/.local/bin/pip shadowing the env's pip)

    echo "--- conda install cuda ${CUDA_VERSION} toolchain ---"
    conda install -y -c nvidia \
        "cuda-version=${CUDA_VERSION}" "cuda-toolkit=${CUDA_VERSION}" \
        cusparselt nccl cuda-cupti

    _export_cuda_env "${CONDA_PREFIX}"
    # conda uses lib/ + include/ where a system CUDA install uses lib64/ and
    # targets/…; without these the extension builds below fail to find
    # libcudart / cuda_runtime.h.
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH:-}"
    export CPATH="${CONDA_PREFIX}/include:${CPATH:-}"

    # A stale CUDA_HOME/PATH from an earlier shell or module would silently build
    # the CUDA extensions below against the wrong toolkit.
    NVCC_VERSION="$("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    if [ "${NVCC_VERSION}" != "${CUDA_VERSION}" ]; then
        echo "ERROR: nvcc at ${CUDA_HOME}/bin/nvcc reports version ${NVCC_VERSION}, expected ${CUDA_VERSION}." >&2
        echo "Check for a stale CUDA_HOME/PATH, or run: conda list -p ${CONDA_ENV_PATH} cuda-toolkit" >&2
        exit 1
    fi
    echo "--- CUDA_HOME=${CUDA_HOME} ---"
else
    if [ ! -d "${VENV_PATH}" ]; then
        echo "--- python${PYTHON_VERSION} -m venv ${VENV_PATH} ---"
        "python${PYTHON_VERSION}" -m venv "${VENV_PATH}"
    fi
    echo "--- activate ${VENV_PATH} ---"
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
fi

echo "--- pip bootstrap ---"
# ninja is not optional in practice: without it torch's BuildExtension falls back
# to the distutils backend, which prepends the interpreter's sysconfig CFLAGS to
# the *nvcc* command line. Under conda those are the conda-toolchain flags
# (-fno-plt, -ftree-vectorize, -fdebug-prefix-map, …), which nvcc rejects with
# "nvcc fatal: Unknown option", breaking every CUDA extension build below.
pip install --upgrade pip setuptools wheel ninja
# Build backends for the dependencies that have no aarch64 wheel and so must be
# compiled from source: libigl needs scikit_build_core + cmake, gdist needs
# Cython + numpy. o3b is installed with --no-build-isolation, so pip cannot pull
# these in itself -- without them the install dies with "Cannot import
# 'scikit_build_core.build'" / "No module named 'Cython'".
# No-ops on x86_64, where those dependencies all resolve to wheels.
pip install --upgrade cmake scikit-build-core pybind11 Cython numpy setuptools-scm

# ensure correct version -- pinned, not "latest in the index". TORCH_VERSION
# already names the env (venv_..._torch26) and builds the pytorch3d wheel tag
# (pt2.6.0cu126) and the torch-scatter URLs below, so an unpinned install that
# resolves to something else yields an env whose name lies and whose extension
# wheels are built for the wrong torch ABI. Unpinned happened to give 2.6.0 on
# the cu124 index but 2.13.0 on cu126.
pip install "torch==${TORCH_VERSION}" torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# nvcc rejects the default host compiler on ubuntu 24 (gcc-14); pin it to 13.
# Guarded so nodes without gcc-13 keep their default compiler. Stays exported
# for the CUDA extension builds further below (pointnet2, Mask2Former, sm-mrf).
if [ -x /usr/bin/gcc-13 ] && [ -x /usr/bin/g++-13 ]; then
    export CC=/usr/bin/gcc-13    # only required for ubuntu 24
    export CXX=/usr/bin/g++-13   # only required for ubuntu 24
    echo "--- host compiler pinned: CC=${CC} CXX=${CXX} ---"
fi
python -m pip install nvdiffrast@git+https://github.com/NVlabs/nvdiffrast --no-build-isolation

# Compiled extensions link against the exact libtorch they were built with, but
# pip sees them as "already satisfied" and skips them when torch later changes
# version. The stale .so then fails at *run* time with an undefined c10 symbol
# (e.g. _ZN3c104cuda29c10_cuda_check_implementation...), the method fails to
# build, and run.py silently degrades the run to GT/oracle. Import each one and
# rebuild the ones that no longer load.
_rebuild_if_broken() {
    local module="$1"; shift
    if python -c "import ${module}" >/dev/null 2>&1; then
        return 0
    fi
    echo "--- ${module} does not import against torch ${TORCH_VERSION}; rebuilding ---"
    python -m pip install --no-build-isolation --force-reinstall --no-cache-dir "$@"
    if python -c "import ${module}" >/dev/null 2>&1; then
        echo "    ${module} rebuilt OK"
    else
        echo "WARNING: ${module} still does not import -- methods needing it will"
        echo "         fail to build and the run will fall back to GT/oracle."
    fi
}

_rebuild_if_broken nvdiffrast.torch \
    "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast"


#python -m pip install --no-build-isolation visdom
#python -m pip install --no-build-isolation "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast"


echo "--- pip install o3b ---"
pip install -e third_party/o3b --no-build-isolation

echo "--- pip install housecorr3d ---"
pip install -e .

# ── torch.hub cache warm-up ───────────────────────────────────────────────────
# torch.hub.load() downloads on first use. JSC compute nodes have no internet,
# so that fails at run time with "no internet connection and the repo could not
# be found in the cache" and the method silently falls back to GT/oracle.
# Populate the cache here on the login node instead; TORCH_HOME points at shared
# storage (JSC homes are quota-tight) so the jobs find it.
# WARM_TORCH_HUB is a space-separated list of "<owner>/<repo>:<model>".
if [ -n "${WARM_TORCH_HUB:-}" ]; then
    echo "--- warming torch.hub cache (TORCH_HOME=${TORCH_HOME:-<default>}) ---"
    for _entry in ${WARM_TORCH_HUB}; do
        _repo="${_entry%%:*}"
        _model="${_entry##*:}"
        if python -c "
import sys, torch
torch.hub.load('${_repo}', '${_model}', pretrained=True)
print('    cached ${_repo} ${_model}')
"; then :; else
            echo "WARNING: could not warm ${_repo} ${_model} -- jobs needing it will fall back to GT/oracle"
        fi
    done
fi

echo "--- pip install optional deps ---"
pip install pyrender2
pip install xatlas

# open3d and libigl are o3b extras rather than hard dependencies: neither can be
# installed on linux-aarch64 (see the note in o3b's pyproject.toml). Installed
# best-effort so x86_64 platforms keep getting them exactly as before, while
# aarch64 carries on without -- nothing on o3b's core import path needs either.
for _opt in open3d libigl; do
    if ! pip install "${_opt}"; then
        echo "WARNING: ${_opt} unavailable for $(uname -m)/python${PYTHON_VERSION} -- continuing."
        echo "         Code paths importing it will fail; on JSC use renderer: pytorch3d."
    fi
done


if [ "${INSTALL_DIFF3F}" = "true" ] || [ "${INSTALL_DIFF3F}" = "True" ]; then
    echo "--- pip install diff3f deps ---"
    _install_pytorch3d_checked
    pip install diffusers transformers accelerate
    pip install git+https://github.com/skoch9/meshplot.git
    pip install pythreejs
    # Recompile gdist from source against the final numpy in this venv.
    # Binary wheels are pinned to the numpy ABI at build time; optional deps
    # (pytorch3d, diffusers, …) can silently shift numpy afterwards, causing
    # "numpy.dtype size changed, may indicate binary incompatibility".
    pip install --no-binary gdist gdist --force-reinstall --no-cache-dir
fi


if [ "${INSTALL_DENSEMATCHER}" = "true" ] || [ "${INSTALL_DENSEMATCHER}" = "True" ]; then
  # pip install --no-cache-dir xformers  # requires torch 2.10 not required
  pip install --no-cache-dir robust-laplacian
  pip install --no-cache-dir potpourri3d
  pip install diffusers[torch]==0.27.2
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/Mask2Former # export CUDA_HOME="/usr/local/cuda-12.4" & 
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/ODISE
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/stablediffusion
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/featup
  pip install --no-build-isolation --no-cache-dir ./third_party/o3b/src/o3b/model/densematcher/third_party/dift
  pip install "numpy<2.0" # ==1.24.1       # < 2.0: numpy.core.multiarray removed in NumPy 2
  pip install "Pillow<10.0.0"    # Image.LINEAR removed in Pillow 10
  pip install setuptools==81.0.0
  pip install pytorch-lightning==1.9.5 kornia==0.7.2 pillow==9.3.0 transformers==4.27.0 matplotlib==3.9.3
  pip install huggingface-hub==0.25.2
fi


if [ "${INSTALL_MORPHEUS}" = "true" ] || [ "${INSTALL_MORPHEUS}" = "True" ]; then
    echo "--- pip install morpheus deps ---"
    _install_pytorch3d_checked
    # The complete GenPose2 setup is handled by the INSTALL_GENPOSE2 block
    # below (forced on by INSTALL_MORPHEUS).
fi


if [ "${INSTALL_GENPOSE2}" = "true" ] || [ "${INSTALL_GENPOSE2}" = "True" ]; then
    GENPOSE2_DIR="third_party/o3b/src/o3b/model/obj_pose/genpose2/third_party"

    echo "--- pip install genpose2 (object-pose estimator) ---"
    # GenPose2 is vendored under the o3b obj_pose model; install it editable so
    # `import genpose2` works for the GenPose2 model / GenPose2Method.
    pip install --no-build-isolation -e "${GENPOSE2_DIR}/GenPose2"

    # GenPose2 imports the PointNet++ CUDA ops package `pointnet2`, which it does
    # NOT declare as a pip dependency and which must be compiled (needs nvcc /
    # the CUDA toolchain). Build + install it (non-editable so the compiled
    # pointnet2_cuda extension lands in site-packages).
    # NOTE: GenPose2 also needs `cutoop` (pulled in as a declared dependency
    # above, but it requires OpenEXR headers: `apt-get install openexr`).
    echo "--- pip install pointnet2 (CUDA ops for GenPose2) ---"
    pip install --no-build-isolation "${GENPOSE2_DIR}/pointnet2"
fi


if [ "${INSTALL_MAGICPONY}" = "true" ] || [ "${INSTALL_MAGICPONY}" = "True" ]; then
    # MagicPony (https://github.com/elliottwu/MagicPony), vendored under
    # od3d's third_party (not a git submodule, checked directly into od3d).
    # Provides `import magicpony` for MagicPonyMethod. nvdiffrast (its
    # renderer) and xatlas are already installed above / declared by o3b.
    echo "--- pip install magicpony deps ---"
    pip install --no-build-isolation -e third_party/od3d/third_party/MagicPony
fi


if [ "${INSTALL_PARTFIELD}" = "true" ] || [ "${INSTALL_PARTFIELD}" = "True" ]; then
    # PartField (https://github.com/nv-tlabs/PartField), feature-extraction
    # subset vendored under o3b/model/partfield/. The Objaverse checkpoint is
    # downloaded at runtime by PartFieldModel (→ checkpoints/partfield/).
    echo "--- pip install partfield deps ---"
    pip install einops
    # pre-built wheel matching the torch+cuda combo, e.g. torch-2.6.0+cu124
    pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
fi


if [ "${INSTALL_GECO}" = "true" ] || [ "${INSTALL_GECO}" = "True" ]; then
    # GeCo: "Fast Globally Optimal and Geometrically Consistent 3D Shape
    # Matching" (Roetzer & Bernard, ICCV 2025), https://github.com/paul0noah/geco
    # Note: gurobi needs a license for larger models (free academic licenses
    # exist); the pip-bundled trial license only solves small problems. Without
    # a license, GeCoMethod falls back to the open-source scipy HiGHS solver.
    echo "--- pip install geco deps ---"
    pip install gurobipy libigl "git+https://github.com/paul0noah/GeCo"
fi


if [ "${INSTALL_MRF3DTOPO}" = "true" ] || [ "${INSTALL_MRF3DTOPO}" = "True" ]; then
    # sm-mrf: "Fast Markov Random Field Optimisation for Topologically Noisy 3D
    # Shape Matching" (Roetzer, Thunberg, Lähner & Bernard, CVPR 2026),
    # https://github.com/paul0noah/sm-mrf
    # Needs push access to the private repo (SSH key on this host), CMake
    # >= 3.16 and a C++17 compiler.
    echo "--- pip install sm-mrf (MRF3DTopoMethod) ---"
    # sm-mrf's setup.py does `import pkg_resources`, which setuptools>=82
    # removed entirely (last version bundling it is 81.0.0).
    pip install libigl "setuptools<82"
    #pip install --no-build-isolation "git+ssh://git@github.com/paul0noah/sm-mrf.git"
    pip install --no-build-isolation "git+https://github.com/paul0noah/sm-mrf.git"
    # Recompile gdist from source against the final numpy in this venv.
    # Binary wheels are pinned to the numpy ABI at build time; optional deps
    # (pytorch3d, diffusers, …) can silently shift numpy afterwards, causing
    # "numpy.dtype size changed, may indicate binary incompatibility".
    pip install --no-binary gdist gdist --force-reinstall --no-cache-dir
fi

echo "=== setup complete ==="
