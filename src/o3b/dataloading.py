"""Torch multiprocessing setup driven by the environment.

Applied once at ``import o3b`` (see ``o3b/__init__.py``) from environment
variables the platform's job preamble exports — so *what* to use is a platform
config decision (``configs/platform/slurm.yaml``), not a per-DataLoader one.

    MP_SHARING_STRATEGY   file_system | file_descriptor
    MP_START_METHOD       fork | spawn | forkserver

Why it matters on slurm: torch's default ``file_descriptor`` strategy sends one
fd per shared tensor from each worker to the consumer process.  Our batches
carry dozens of tensors each (rgb / depth / mask / kpts / mesh per frame object,
twice for a pair) and prefetch keeps several batches in flight, so the receiving
process holds thousands of fds.  The cluster's soft RLIMIT_NOFILE is 1024
(inherited from the submit host — ``PropagateResourceLimits=ALL``), where the
workstation's is 1048576, so only the cluster hits it, as

    RuntimeError: received 0 items of ancdata

in the pin-memory thread, followed by "Pin memory thread exited unexpectedly".
``file_system`` passes /dev/shm names instead of fds and has no such limit; the
job preamble also raises the fd limit itself (``nofile_limit`` in the platform
config), so the two mitigations are independent.

Selecting ``file_system`` also runs `check_shm`, which warns when /dev/shm is
already filling up or still holds segments an earlier killed job leaked — both
surface much later, and far less legibly, as a worker crash.

Kept free of heavy / optional dependencies (unlike o3b.multiprocessing, which
pulls in sqlalchemy) so it is safe to import from the package root.
"""
import logging
import os

logger = logging.getLogger(__name__)

SHM_PATH = "/dev/shm"
SHM_WARN_USED_PCT = 50.0   # override with MP_SHM_WARN_PCT
SHM_WARN_FREE_GB = 4.0     # absolute floor, whatever the percentage says


def _gb(n_bytes: int) -> float:
    return n_bytes / 1024 ** 3


def _fmt_size(n_bytes: int) -> str:
    for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n_bytes >= scale:
            return f"{n_bytes / scale:.1f} {unit}"
    return f"{n_bytes} B"


def _stale_torch_files(path: str, limit: int = 100_000) -> tuple[int, int]:
    """(count, bytes) of this user's torch_* segments in ``path``.

    Called before any worker of *this* process exists, so whatever it finds was
    orphaned by an earlier job — file_system sharing unlinks its segments on a
    clean exit, but not when the process is SIGKILLed (slurm timeout, OOM).
    ``limit`` caps the scan so a pathological directory cannot stall startup.
    """
    uid = os.getuid()
    count = total = 0
    try:
        with os.scandir(path) as entries:
            for i, entry in enumerate(entries):
                if i >= limit:
                    logger.debug(f"{path}: stopped scanning after {limit} entries")
                    break
                if not entry.name.startswith("torch_"):
                    continue
                try:
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue          # vanished mid-scan, or not ours to stat
                if stat.st_uid == uid:
                    count += 1
                    total += stat.st_size
    except OSError as exc:
        logger.debug(f"could not scan {path}: {exc}")
    return count, total


def check_shm(path: str = SHM_PATH) -> str:
    """Warn if shared memory is already full or littered before we rely on it.

    The file_system strategy allocates every shared tensor as a file here, so a
    node left full by other jobs turns into "No space left on device" or a bus
    error inside a worker, well after startup and with no mention of /dev/shm.
    Warning up front makes that diagnosable — and points at the leftovers, which
    are the part the user can actually delete.

    Returns a one-line usage summary ("" if the path is unusable) for callers
    that report it alongside the strategy.
    """
    import shutil

    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        logger.warning(f"file_system sharing strategy, but {path} is not usable: {exc}")
        return ""

    used_pct = 100.0 * usage.used / usage.total if usage.total else 0.0
    stale_count, stale_bytes = _stale_torch_files(path)
    state = (f"{path}: {_gb(usage.used):.1f}/{_gb(usage.total):.1f} GB used "
             f"({used_pct:.0f}%), {_gb(usage.free):.1f} GB free")

    try:
        warn_pct = float(os.environ.get("MP_SHM_WARN_PCT", SHM_WARN_USED_PCT))
    except ValueError:
        warn_pct = SHM_WARN_USED_PCT

    if used_pct >= warn_pct or _gb(usage.free) < SHM_WARN_FREE_GB:
        logger.warning(
            f"shared memory is filling up — {state}. DataLoader workers allocate "
            f"their shared tensors here under the file_system strategy and will "
            f"fail once it is full."
        )
    else:
        logger.info(f"{state} (file_system sharing strategy)")

    if stale_count:
        logger.warning(
            f"{path} holds {stale_count} torch_* segment(s) ({_fmt_size(stale_bytes)}) "
            f"left by an earlier job of yours (killed before it could clean up). "
            f"""Remove with: find {path} -user "$USER" -name 'torch_*' -delete"""
        )
    return state


def apply_mp_env() -> None:
    """Apply MP_SHARING_STRATEGY / MP_START_METHOD if set. No-op otherwise.

    Unset variables leave torch's defaults alone, so nothing changes off the
    cluster unless the environment asks for it.

    Prints what torch ends up with (the CLI configures no logging, so INFO is
    invisible): the job preamble echoes what it *exported*, this reports what
    the process actually runs with — they differ if the value is unknown to
    this torch build, or if the cluster checkout predates the setting.
    """
    strategy = os.environ.get("MP_SHARING_STRATEGY", "")
    start_method = os.environ.get("MP_START_METHOD", "")
    if not strategy and not start_method:
        return

    import torch.multiprocessing as mp

    shm_state = ""
    if strategy:
        if strategy in mp.get_all_sharing_strategies():
            mp.set_sharing_strategy(strategy)
            if strategy == "file_system":
                shm_state = check_shm()
        else:
            logger.warning(f"MP_SHARING_STRATEGY={strategy!r} unknown, keeping "
                           f"{mp.get_sharing_strategy()!r}")
    if start_method:
        if start_method in mp.get_all_start_methods():
            # force: torch/other imports may have set one already
            mp.set_start_method(start_method, force=True)
        else:
            logger.warning(f"MP_START_METHOD={start_method!r} unknown, keeping "
                           f"{mp.get_start_method()!r}")

    # allow_none: plain get_start_method() would *fix* the context's start
    # method as a side effect of asking
    summary = (f"sharing_strategy={mp.get_sharing_strategy()} "
               f"start_method={mp.get_start_method(allow_none=True) or 'default'}")
    print(f"MP:      {summary}" + (f"  ({shm_state})" if shm_state else ""), flush=True)
