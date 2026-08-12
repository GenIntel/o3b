"""Duplicate a dataset's ``tform_obj_type`` directory under a new name.

    o3b dataset cp-tform-obj-type -d every9d_v3 -t every9d_v5

copies

    <path_preprocess>/tform_obj/all_canonical_poses_filtered_center3d_auto/
 -> <path_preprocess>/tform_obj/every9d_v5/

so a new labelling round can start from an existing one instead of from
nothing.  The whole subtree is copied verbatim, ``meta_mask/meta/<category>/
<sequence>/tform_obj.pt`` and all.

The target name doubles as a dataset name: the copy is followed by writing
``configs/dataset/<target>.yaml``, a config inheriting from the ``-d`` one with
``tform_obj_type`` pointed at the new directory, so ``o3b dataset viz -d
every9d_v5`` works immediately.  ``--no-config`` skips that half.

Both sides live under the same ``path_preprocess``, so the copy is always
single-host: with ``-p slurm`` it runs *on* the cluster (a native rsync over
~114k tiny files, a minute), and only without a remote platform does it fall
back to running here — which over an sshfs mount is file-by-file and slow
enough to be worth warning about.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

_TFORM_OBJ_PARENT = "tform_obj"


def tform_obj_dir(path_preprocess: Path, tform_obj_type: str) -> Path:
    return Path(path_preprocess) / _TFORM_OBJ_PARENT / tform_obj_type


def _run(host: str | None, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command here, or on *host* when the platform is remote."""
    argv = (["ssh", "-q", "-o", "BatchMode=yes", host, command] if host
            else ["bash", "-c", command])
    proc = subprocess.run(argv, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed on {host or 'localhost'}: {command}\n{proc.stderr.strip()}"
        )
    return proc


def _exists(host: str | None, path: Path) -> bool:
    return _run(host, f"test -d {shlex.quote(str(path))}", check=False).returncode == 0


def _exists_file(host: str | None, path: Path) -> bool:
    return _run(host, f"test -f {shlex.quote(str(path))}", check=False).returncode == 0


def _count(host: str | None, path: Path) -> int:
    out = _run(host, f"find {shlex.quote(str(path))} -name tform_obj.pt | wc -l").stdout
    return int(out.strip() or 0)


def _count_db(host: str | None, path: Path) -> int:
    """Rows in a tform_obj .db, counted on whichever machine holds it.

    Through python3's sqlite3 module rather than the sqlite3 CLI, which is not
    installed on either the workstation or the cluster login node.
    """
    if host is None:
        import sqlite3
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
            try:
                return con.execute("SELECT COUNT(*) FROM tform_obj").fetchone()[0]
            finally:
                con.close()
        except Exception:
            return -1
    snippet = (
        "import sqlite3,sys;"
        f"c=sqlite3.connect('file:{path}?mode=ro',uri=True);"
        "print(c.execute('SELECT COUNT(*) FROM tform_obj').fetchone()[0])"
    )
    out = _run(host, f"python3 -c {shlex.quote(snippet)}", check=False).stdout.strip()
    return int(out) if out.isdigit() else -1


def _looks_like_sshfs(path: Path) -> bool:
    """True when *path* sits under a fuse mount on this machine."""
    import os

    p = Path(path)
    for cand in [p, *p.parents]:
        if os.path.ismount(str(cand)):
            try:
                with open("/proc/mounts") as f:
                    return any(line.split()[1] == str(cand) and "fuse" in line.split()[2]
                               for line in f if len(line.split()) > 2)
            except OSError:
                return False
    return False


def run_tform_obj_to_db(config_path: Path, *, platform: str = "default",
                        override: bool = False, workers: int = 32) -> None:
    """`o3b dataset tform-obj-to-db -d <dataset>` — fold the directory into a .db."""
    import time

    from o3b.dataset import tform_obj_store as store
    from o3b.dataset.cli import _platform_to_dataset_overrides
    from o3b.dataset.dataset import DatasetConfig

    cfg = DatasetConfig.from_yaml(Path(config_path),
                                  overrides=_platform_to_dataset_overrides(platform))
    if cfg.tform_obj_type == "raw":
        print(f"{Path(config_path).name} has tform_obj_type: raw — nothing on disk to "
              f"convert.", file=sys.stderr)
        sys.exit(1)
    if not cfg.path_preprocess:
        print(f"platform '{platform}' resolves no path_preprocess", file=sys.stderr)
        sys.exit(1)

    src = store.dir_path(cfg.path_preprocess, cfg.tform_obj_type)
    out = store.db_path(cfg.path_preprocess, cfg.tform_obj_type)
    print(f"Converting {cfg.tform_obj_type}:")
    print(f"  from  {src}/")
    print(f"  to    {out}")
    if _looks_like_sshfs(src):
        print(f"\n  NOTE: that is an sshfs mount, so this reads ~100k files over the\n"
              f"        network (tens of minutes). Run it where the data lives:\n"
              f"        o3b dataset tform-obj-to-db -d {Path(config_path).stem} "
              f"-p slurm_lmbl40 --remote")

    t0 = time.time()
    try:
        n, path = store.convert_dir_to_db(cfg.path_preprocess, cfg.tform_obj_type,
                                          override=override, workers=workers)
    except (FileNotFoundError, FileExistsError) as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)
    dt = time.time() - t0
    size = path.stat().st_size
    print(f"\n  {n} transforms -> {path}  ({size / 1e6:.1f} MB) in {dt:.1f}s")
    print(f"\nDone. The loader now prefers it over {src.name}/ — "
          f"delete that directory once you trust the db.")


def _dataset_configs_dir(config_path: Path) -> Path:
    """Where to write the new dataset yaml.

    Next to the ``-d`` config when that lives in the standard configs/dataset
    directory (so ``-d <name>`` resolves the new one too); otherwise beside
    whatever path was passed, which is where its ``defaults:`` parent sits.
    """
    from o3b.dataset import cli as _cli

    standard = (Path(_cli.__file__).parent.parent.parent / "configs" / "dataset").resolve()
    return standard if Path(config_path).resolve().parent == standard else Path(config_path).parent


def _config_text(target: str, source_stem: str, source_type: str, platform: str) -> str:
    from datetime import date

    return f"""\
# {target} — {source_stem} under the canonical poses in tform_obj/{target}/.
#
# Forked from {source_type}
# by `o3b dataset cp-tform-obj-type -d {source_stem} -t {target} -p {platform}`
# on {date.today().isoformat()}, and free to diverge from it: this config reads
# <path_preprocess>/tform_obj/{target}/, so relabelling or deleting a
# tform_obj.pt there changes what this dataset contains (a sequence without one
# is skipped).
#
#   o3b dataset viz -d {target} -c chair
#
# Everything else — paths, mesh_type, modalities, transform, the frame/sequence
# caps — is inherited from {source_stem}.
#
# NOTE: od3d keys some of its behaviour off a `_center3d_auto` suffix in this
# name; o3b does not, and filters purely on the tform_obj.pt existing.

defaults:
  - {source_stem}
  - _self_

tform_obj_type: "{target}"
"""


def write_dataset_config(
    config_path: Path,
    target: str,
    source_type: str,
    *,
    platform: str = "default",
    dry_run: bool = False,
    override: bool = False,
) -> Path | None:
    """Write configs/dataset/<target>.yaml inheriting from the ``-d`` config."""
    out = _dataset_configs_dir(config_path) / f"{target}.yaml"
    source_stem = Path(config_path).stem

    if out.exists() and not override:
        print(f"\n  dataset config already exists, leaving it alone: {out}\n"
              f"  (pass --override to rewrite it)")
        return None

    text = _config_text(target, source_stem, source_type, platform)
    verb = "Would write" if dry_run else ("Rewriting" if out.exists() else "Writing")
    print(f"\n  {verb} dataset config: {out}")
    if dry_run:
        print("".join(f"      {line}\n" for line in text.splitlines()))
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out


def copy_tform_obj_type(
    config_path: Path,
    target: str,
    *,
    platform: str = "default",
    dry_run: bool = False,
    override: bool = False,
    write_config: bool = True,
) -> None:
    from o3b.dataset.cli import _platform_to_dataset_overrides
    from o3b.dataset.dataset import DatasetConfig
    from o3b.dataset.sync import _ssh_host

    cfg = DatasetConfig.from_yaml(Path(config_path),
                                  overrides=_platform_to_dataset_overrides(platform))
    if not cfg.path_preprocess:
        print(f"platform '{platform}' resolves no path_preprocess for {config_path}",
              file=sys.stderr)
        sys.exit(1)
    if cfg.tform_obj_type == "raw":
        print(
            f"{Path(config_path).name} has tform_obj_type: raw — that is the dataset's own\n"
            f"object frame, not a directory on disk, so there is nothing to copy.\n"
            f"Point -d at a config with a canonical tform_obj_type (e.g. every9d_v3).",
            file=sys.stderr,
        )
        sys.exit(1)
    if "/" in target or target in ("", ".", ".."):
        print(f"invalid target {target!r}: it names a single subdirectory of "
              f"{_TFORM_OBJ_PARENT}/, not a path", file=sys.stderr)
        sys.exit(1)
    if target == cfg.tform_obj_type:
        print(f"target {target!r} is the source — nothing to do.", file=sys.stderr)
        sys.exit(1)

    from o3b.dataset import tform_obj_store as store

    host = _ssh_host(platform)
    src_db = store.db_path(cfg.path_preprocess, cfg.tform_obj_type)
    dst_db = store.db_path(cfg.path_preprocess, target)
    src_dir = tform_obj_dir(cfg.path_preprocess, cfg.tform_obj_type)
    dst_dir = tform_obj_dir(cfg.path_preprocess, target)

    # A .db is one 10 MB file, so forking it is a copy rather than a 114k-file
    # rsync — seconds instead of minutes. Fall back to the directory layout when
    # the source has not been converted.
    use_db = not dry_run and _exists_file(host, src_db)
    if dry_run:
        use_db = _exists_file(host, src_db)
    src, dst = (src_db, dst_db) if use_db else (src_dir, dst_dir)

    where = f"on {host}" if host else "locally"
    kind = "one .db file" if use_db else "per-sequence directory"
    print(f"Copying tform_obj_type {where} ({platform}) — {kind}:")
    print(f"  from  {src}")
    print(f"  to    {dst}")

    if not dry_run and not (_exists_file(host, src) if use_db else _exists(host, src)):
        print(f"\nsource does not exist: {src}", file=sys.stderr)
        sys.exit(1)
    if not dry_run and (_exists_file(host, dst) if use_db else _exists(host, dst)) and not override:
        print(f"\ntarget already exists: {dst}\n"
              f"Pass --override to replace it.", file=sys.stderr)
        sys.exit(1)

    if host is None and not use_db and _looks_like_sshfs(src):
        print(f"\n  NOTE: {src} is an sshfs mount, so every one of the ~100k files\n"
              f"        would be copied over the network twice. Run it on the machine\n"
              f"        that owns the data instead:  -p slurm\n"
              f"        (or convert it first: o3b dataset tform-obj-to-db)")

    if use_db:
        cmd = f"cp -f {shlex.quote(str(src))} {shlex.quote(str(dst))}"
    else:
        # Single host, two local paths — rsync so the copy is restartable and
        # shows progress; a plain `cp -r` gives neither over a tree this size.
        cmd = (f"mkdir -p {shlex.quote(str(dst))} && "
               f"rsync -a --partial --info=progress2 "
               f"{shlex.quote(str(src))}/ {shlex.quote(str(dst))}/")
    argv = (["ssh", "-q", host, cmd] if host else ["bash", "-c", cmd])
    print("\n  " + (" ".join(shlex.quote(a) for a in argv) if host else cmd))
    if dry_run:
        if write_config:
            write_dataset_config(config_path, target, cfg.tform_obj_type,
                                 platform=platform, dry_run=True, override=override)
        print("\n(dry run — nothing copied, nothing written)")
        return

    if subprocess.run(argv).returncode != 0:
        print("rsync failed", file=sys.stderr)
        sys.exit(1)

    if use_db:
        n_src, n_dst = _count_db(host, src), _count_db(host, dst)
        unit = "rows"
    else:
        n_src, n_dst = _count(host, src), _count(host, dst)
        unit = "tform_obj.pt"
    print(f"\n  {n_src} {unit} in source, {n_dst} in target")
    if n_src < 0 or n_dst < 0:
        print("  (could not verify — sqlite3 CLI not available on that machine)")
        n_src = n_dst = 0
    if n_dst < n_src:
        # Leave the config unwritten: a dataset name that resolves to a
        # half-copied directory is worse than no dataset name at all.
        print(f"  WARNING: target is short by {n_src - n_dst} files — re-run to resume.",
              file=sys.stderr)
        sys.exit(1)

    written = (write_dataset_config(config_path, target, cfg.tform_obj_type,
                                    platform=platform, override=override)
               if write_config else None)
    if written is not None:
        print(f"\nDone. Use it with  o3b dataset viz -d {target}")
    else:
        print(f"\nDone. Use it with  tform_obj_type: {target}  in a dataset config.")
