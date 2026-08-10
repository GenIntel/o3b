"""Copy a materialised sharded dataset cache from one platform to another.

    o3b dataset sync-shard -d hc3d_frame_object_train_cat_sharded -s slurm -t slurm_jupiter

A sharded cache lives at ``<path_datasets_preprocess>/sharded/<sharded_name>``
(see ``ConfigurableDataset._sharded_dir``).  Both platforms resolve the *same*
dataset config, so only the ``path_datasets_preprocess`` prefix differs — the
directory name is identical on both sides and is used to pair source and target.

WHERE THIS RUNS, AND WHY
------------------------
On the *local* machine, always — it is the only host that can reach both
clusters.  Measured (2026-08):

  * ``slurm`` (kislogin1.rz.ki.privat) → ``jupiter``: BLOCKED.  The LMB nodes
    have no direct route out; everything outbound goes through the HTTP proxy
    (``http_proxy`` in configs/platform/slurm.yaml), which does not carry ssh.
  * ``jupiter`` → ``slurm``: BLOCKED.  kislogin1 is an RZ-internal ``.privat``
    name that does not even resolve from Jülich.
  * JSC additionally demands a TOTP token on *every* new connection.  Only the
    local ``~/.ssh/config`` can answer it, once, via ControlMaster — a push from
    a cluster could not authenticate even if the route existed.

So a source-side push or a target-side pull is impossible, and rsync refuses
remote→remote anyway ("The source and destination cannot both be remote").
The data is therefore streamed *through* this machine with a single pipe

    ssh SRC 'cd <dir> && tar -cf - --files-from=-'  |  ssh TGT 'cd <stage> && tar -xf -'

which never stages anything on local disk and costs one pass, not two.
Measured end-to-end on that exact pipe: ~140 MB/s (slurm → local → jupiter),
i.e. ~65 s for a 9 GB single-category shard directory.

Compression is off by default and ``--compress`` is rarely worth it: the shard
payload is already zstd-compressed (v3 codec), ``zstd -1`` only reaches ~0.69 on
top of it, and at 140 MB/s the pipe is not the bottleneck — a measured
compressed relay came out *slower* than the plain one.

When one of the two platforms is local (``ssh: False``, e.g. ``-s default``),
there is no relay to build and plain ``rsync`` is used instead: it resumes
partial *files*, which the tar pipe cannot.

INTERRUPTION SAFETY
-------------------
Nothing is written to the final target path until the whole directory is there:
files land in ``<target>.insync`` and are moved into place with a single ``mv``.
An interrupted run leaves that staging directory behind, and the next run
transfers only the files it still lacks (compared by relative path and size —
shard files are immutable once written), so re-running resumes rather than
restarts.  A target that already exists is left alone unless ``--override``.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Files are transferred in batches of roughly this many bytes; each batch is one
# tar pipeline, which is what gives progress reporting (and a resume point).
_BATCH_BYTES = 2 << 30

# Suffix of the staging directory next to the final target path.
_STAGE_SUFFIX = ".insync"

# Guard for the `rm -rf` in _finalise: every path we build ends in .../sharded/<name>.
_SHARDED_PARENT = "sharded"

_SAFE_GLOB = re.compile(r"^[A-Za-z0-9._*?\[\]-]+$")


# ── platform / config resolution ──────────────────────────────────────────────

def _ssh_host(platform: str) -> str | None:
    """The platform's ssh host, or None when it is this machine."""
    from o3b.cli import _load_platform_config

    cfg, _ = _load_platform_config(platform)
    host = cfg.get("ssh")
    if not host or str(host).lower() in ("false", "none", ""):
        return None
    return str(host)


def _sharded_location(config_path: Path, platform: str, categories: str | None) -> tuple[Path, str]:
    """(<path_preprocess>/sharded, sharded_name) for *platform*.

    ``sharded_name`` may contain glob characters — ``-c '*'`` resolves
    ``${category}`` to ``*`` and so selects every per-category directory.
    """
    from o3b.cli import _categories_to_dataset_overrides
    from o3b.dataset.cli import _platform_to_dataset_overrides
    from o3b.dataset.dataset import DatasetConfig

    overrides = (_platform_to_dataset_overrides(platform)
                 + _categories_to_dataset_overrides(categories))
    cfg = DatasetConfig.from_yaml(Path(config_path), overrides=overrides)
    if not cfg.sharded_name:
        raise ValueError(
            f"{Path(config_path).name} has no 'sharded_name' — it has no sharded "
            f"cache to sync. Use a *_sharded config."
        )
    if not cfg.path_preprocess:
        raise ValueError(f"platform '{platform}' resolves no path_preprocess for {config_path}")
    return Path(cfg.path_preprocess) / _SHARDED_PARENT, str(cfg.sharded_name)


# ── remote/local primitives ───────────────────────────────────────────────────

def _ssh(host: str, command: str) -> list[str]:
    """ssh argv for a data-carrying command (stdin/stdout are the payload).

    BatchMode: neither end can answer a prompt here — stdin is the file list or
    the tar stream — so fail fast instead of hanging. `_preflight` opens an
    interactive connection first, which is where a TOTP token gets entered.
    """
    return ["ssh", "-q", "-o", "BatchMode=yes", "-e", "none", host, command]


def _preflight(host: str) -> None:
    """Open one interactive connection so ssh can ask for a password / TOTP.

    On jupiter this also primes the ControlMaster socket, after which every
    BatchMode connection below rides on it silently.
    """
    # -q only silences the login banner; an auth prompt still reaches the user.
    if subprocess.run(["ssh", "-q", host, "true"]).returncode != 0:
        raise RuntimeError(
            f"cannot reach ssh host '{host}'. For JSC hosts, authenticate once with "
            f"`ssh {host}` (TOTP) — later connections reuse that master socket."
        )


def _run_capture(host: str | None, command: str) -> str:
    """Run a shell command here or on *host*, returning stdout."""
    argv = _ssh(host, command) if host else ["bash", "-c", command]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        where = host or "localhost"
        raise RuntimeError(f"command failed on {where}: {command}\n{proc.stderr.strip()}")
    return proc.stdout


def _expand_dirs(host: str | None, root: Path, pattern: str) -> list[str]:
    """Directory names under *root* matching *pattern* (no glob chars = itself)."""
    if not _SAFE_GLOB.match(pattern):
        raise ValueError(f"refusing to expand unsafe sharded_name pattern: {pattern!r}")
    if not any(ch in pattern for ch in "*?["):
        return [pattern]

    # the pattern must stay unquoted for the remote shell to expand it; it was
    # whitelisted above. nullglob keeps an unmatched pattern from echoing itself.
    listing = _run_capture(
        host,
        f"shopt -s nullglob; for d in {shlex.quote(str(root))}/{pattern}/; do "
        f'[ -d "$d" ] && basename "$d"; done',
    )
    names = [n for n in listing.split() if _STAGE_SUFFIX not in n and ".trash-" not in n]
    return sorted(names)


def _list_files(host: str | None, directory: Path) -> dict[str, int] | None:
    """{relative path: size} for every file below *directory*, None if it is absent."""
    if host is None:
        if not directory.is_dir():
            return None
        out = {}
        for dirpath, _, filenames in os.walk(directory):
            for name in filenames:
                p = Path(dirpath) / name
                out[str(p.relative_to(directory))] = p.stat().st_size
        return out

    d = shlex.quote(str(directory))
    listing = _run_capture(
        host,
        f'if [ -d {d} ]; then cd {d} && find . -type f -printf "%s\\t%P\\n"; '
        f"else echo __MISSING__; fi",
    )
    if listing.startswith("__MISSING__"):
        return None
    out = {}
    for line in listing.splitlines():
        if not line.strip():
            continue
        size, _, rel = line.partition("\t")
        out[rel] = int(size)
    return out


def _human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


# ── transfer ──────────────────────────────────────────────────────────────────

def _tar_relay(src_host: str, src_dir: Path, dst_host: str, dst_dir: Path,
               rels: list[str], compress: bool) -> None:
    """Stream *rels* from src_dir to dst_dir through this machine in one pass."""
    src_cmd = f"cd {shlex.quote(str(src_dir))} && tar -cf - --files-from=-"
    dst_cmd = f"mkdir -p {shlex.quote(str(dst_dir))} && cd {shlex.quote(str(dst_dir))} && tar -xf -"
    if compress:
        src_cmd += " | zstd -1 -T4 -c"
        dst_cmd = (f"mkdir -p {shlex.quote(str(dst_dir))} && cd {shlex.quote(str(dst_dir))} "
                   f"&& zstd -dc | tar -xf -")

    sender = subprocess.Popen(_ssh(src_host, src_cmd), stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE)
    receiver = subprocess.Popen(_ssh(dst_host, dst_cmd), stdin=sender.stdout)
    sender.stdout.close()  # the receiver owns the read end now
    try:
        sender.stdin.write(("\n".join(rels) + "\n").encode())
        sender.stdin.close()
    except BrokenPipeError:
        pass  # the receiver died; the returncodes below carry the real error
    rc_recv = receiver.wait()
    rc_send = sender.wait()
    if rc_send != 0 or rc_recv != 0:
        raise RuntimeError(f"transfer failed (tar exit {rc_send}, untar exit {rc_recv})")


def _rsync(src_host: str | None, src_dir: Path, dst_host: str | None, dst_dir: Path) -> None:
    """rsync a whole directory when at most one side is remote."""
    src = f"{src_host}:{src_dir}/" if src_host else f"{src_dir}/"
    dst = f"{dst_host}:{dst_dir}/" if dst_host else f"{dst_dir}/"
    if dst_host is None:
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
    argv = ["rsync", "-a", "--partial", "--info=progress2", src, dst]
    print("  " + " ".join(shlex.quote(a) for a in argv))
    if subprocess.run(argv).returncode != 0:
        raise RuntimeError("rsync failed")


def _remove_files(host: str | None, directory: Path, rels: list[str]) -> None:
    d = shlex.quote(str(directory))
    command = f"cd {d} && xargs -0 rm -f"
    argv = _ssh(host, command) if host else ["bash", "-c", command]
    subprocess.run(argv, input=b"".join(r.encode() + b"\0" for r in rels), check=True)


def _finalise(host: str | None, stage: Path, target: Path) -> None:
    """Replace *target* with *stage* — the only moment the final path changes."""
    if target.name.endswith(_STAGE_SUFFIX) or target.parent.name != _SHARDED_PARENT:
        raise RuntimeError(f"refusing to replace a path outside .../{_SHARDED_PARENT}/: {target}")
    s, t = shlex.quote(str(stage)), shlex.quote(str(target))
    command = f"if [ -e {t} ]; then rm -rf {t}; fi && mv {s} {t}"
    argv = _ssh(host, command) if host else ["bash", "-c", command]
    if subprocess.run(argv).returncode != 0:
        raise RuntimeError(f"could not move {stage} → {target}")


# ── driver ────────────────────────────────────────────────────────────────────

def _sync_one(name: str, src_host: str | None, src_root: Path,
              dst_host: str | None, dst_root: Path,
              override: bool, dry_run: bool, compress: bool) -> bool:
    """Sync one shard directory. Returns True if anything was transferred."""
    src_dir = src_root / name
    dst_dir = dst_root / name
    stage = dst_root / f"{name}{_STAGE_SUFFIX}"

    src_files = _list_files(src_host, src_dir)
    if src_files is None:
        raise RuntimeError(f"source directory does not exist: {src_host or 'local'}:{src_dir}")
    total = sum(src_files.values())
    print(f"\n── {name} ──")
    print(f"  source {src_host or 'local'}:{src_dir}  ({len(src_files)} files, {_human(total)})")
    print(f"  target {dst_host or 'local'}:{dst_dir}")

    dst_files = _list_files(dst_host, dst_dir)
    if dst_files is not None:
        if dst_files == src_files:
            print("  target is already identical — nothing to do.")
            return False
        if not override:
            missing = len(set(src_files) - set(dst_files))
            print(f"  ERROR: target exists and differs ({missing} file(s) missing, "
                  f"{len(dst_files)} present). Pass --override to replace it.", file=sys.stderr)
            raise RuntimeError(f"target exists and differs: {dst_dir}")
        print("  target exists and differs — will be replaced (--override).")

    # resume: whatever a previous interrupted run already put in the staging dir
    staged = _list_files(dst_host, stage) or {}
    todo = sorted(rel for rel, size in src_files.items() if staged.get(rel) != size)
    extra = sorted(set(staged) - set(src_files))
    todo_bytes = sum(src_files[rel] for rel in todo)
    if staged:
        print(f"  resuming: {len(staged) - len(extra)}/{len(src_files)} file(s) already staged")

    if dry_run:
        print(f"  would transfer {len(todo)} file(s), {_human(todo_bytes)}"
              f"{f' and drop {len(extra)} stale staged file(s)' if extra else ''}")
        if todo and src_host and dst_host:
            print(f"  via: ssh {src_host} 'cd {src_dir} && tar -cf - --files-from=-'"
                  f" | ssh {dst_host} 'cd {stage} && tar -xf -'")
        return bool(todo)

    if extra:
        print(f"  dropping {len(extra)} stale file(s) from the staging directory")
        _remove_files(dst_host, stage, extra)

    if todo:
        if src_host and dst_host:
            batches: list[list[str]] = [[]]
            batch_bytes = 0
            for rel in todo:
                if batch_bytes and batch_bytes + src_files[rel] > _BATCH_BYTES:
                    batches.append([])
                    batch_bytes = 0
                batches[-1].append(rel)
                batch_bytes += src_files[rel]
            done = 0
            for i, batch in enumerate(batches, 1):
                size = sum(src_files[rel] for rel in batch)
                t0 = time.time()
                _tar_relay(src_host, src_dir, dst_host, stage, batch, compress)
                dt = max(time.time() - t0, 1e-6)
                done += size
                print(f"  [{i}/{len(batches)}] {_human(size)} in {dt:.1f}s "
                      f"({size / dt / 1e6:.0f} MB/s) — {_human(done)}/{_human(todo_bytes)}")
        else:
            # one side is local: rsync does its own diffing and resumes files
            _rsync(src_host, src_dir, dst_host, stage)

    final = _list_files(dst_host, stage) or {}
    if final != src_files:
        raise RuntimeError(
            f"staged copy at {stage} does not match the source "
            f"({len(final)} vs {len(src_files)} files) — staging kept, re-run to resume"
        )
    _finalise(dst_host, stage, dst_dir)
    print(f"  done → {dst_dir}")
    return True


def sync_sharded(config_path: Path, source_platform: str, target_platform: str,
                 categories: str | None = None, override: bool = False,
                 dry_run: bool = False, compress: bool = False) -> None:
    """Copy the sharded cache of *config_path* between two platforms."""
    src_root, src_name = _sharded_location(config_path, source_platform, categories)
    dst_root, dst_name = _sharded_location(config_path, target_platform, categories)
    if src_name != dst_name:
        raise RuntimeError(
            f"sharded_name differs between platforms ({src_name} vs {dst_name}); "
            f"the platform config must only override the dataset paths."
        )

    src_host = _ssh_host(source_platform)
    dst_host = _ssh_host(target_platform)
    if src_host is None and dst_host is None:
        raise RuntimeError(
            f"both '{source_platform}' and '{target_platform}' are this machine "
            f"(ssh: False) — nothing to sync between."
        )
    for host in (src_host, dst_host):
        if host:
            _preflight(host)

    names = _expand_dirs(src_host, src_root, src_name)
    if not names:
        raise RuntimeError(f"no shard directory matches {src_root}/{src_name}")
    print(f"Syncing {len(names)} shard directory/ies: {source_platform} → {target_platform}")

    changed = 0
    for name in names:
        if _sync_one(name, src_host, src_root, dst_host, dst_root,
                     override=override, dry_run=dry_run, compress=compress):
            changed += 1
    verb = "would transfer" if dry_run else "transferred"
    print(f"\n{verb} {changed}/{len(names)} shard directory/ies.")
