"""Mount a remote platform's dataset directories over sshfs.

Some datasets only exist on a cluster (UCO3D is ~30 TB of video), but the
viewers, the tform tool and a small-scale `init` all want to read them from the
workstation.  ``o3b dataset sshfs -d uco3d -p slurm`` mounts exactly the
directories the dataset config names — nothing else of the remote tree — so a
local run can resolve them.

The mount points come from the dataset config *without* the platform override,
i.e. from whatever ``path_raw`` / ``path_preprocess`` it resolves to locally;
the sources come from the same config *with* the override.  A dataset that is
also present locally therefore keeps its local paths untouched — see
``configs/dataset/uco3d.yaml``, which points its local paths at
``path_datasets_sshfs`` (``~/.o3b/sshfs/datasets`` by default) precisely so the
mount cannot land on a half-downloaded local copy.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Keep the mount responsive and self-healing: reconnect after a dropped ssh
# connection, and cache attributes/directory listings aggressively — a listing
# over sshfs costs a round trip per entry, and walking a dataset's meta tree is
# nothing but listings.  Mounted read-write by default because `o3b dataset
# index` and `o3b dataset init` write their caches under path_preprocess;
# --read-only trades that away for a guard against touching the remote.
_SSHFS_OPTIONS = [
    "reconnect",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=3",
    "cache=yes",
    "cache_timeout=3600",
    "kernel_cache",
    "entry_timeout=3600",
    "attr_timeout=3600",
    "compression=no",
    # The bulk reads here are already-compressed video, and aes128-gcm is the
    # cheapest cipher openssh offers on hardware with AES-NI.
    "Ciphers=aes128-gcm@openssh.com",
]

# Without this a mount carries every request over one ssh channel, so threads
# gain nothing: eight cold 1 KB reads over the UCO3D mount measured 6.4 s
# serially and 7.0 s across eight threads.  A cold page of the axes editors is
# ~80 such round trips, and everything that fetches them — collect_frames'
# read-ahead and decode pools, the prefetcher — is written to overlap.  N is
# ssh connections, not FUSE threads: each is a real login, so this is also the
# number the remote's MaxSessions and any per-user connection limit see.
_SSHFS_CONNS = 8

_PARALLEL_OPTION = "max_conns"


def _sshfs_supports(option: str) -> bool:
    """Does the installed sshfs know ``-o <option>``? (``max_conns`` is 3.0+)

    Probed rather than assumed: an unknown -o makes sshfs refuse to mount at
    all, which would trade a working slow mount for no mount.
    """
    try:
        out = subprocess.run(["sshfs", "--help"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return option in (out.stdout + out.stderr)


def sshfs_options(read_only: bool = False, conns: int = _SSHFS_CONNS) -> list[str]:
    """The -o list a mount is made with."""
    options = list(_SSHFS_OPTIONS)
    if conns > 1 and _sshfs_supports(_PARALLEL_OPTION):
        options.append(f"{_PARALLEL_OPTION}={conns}")
    return options + (["ro"] if read_only else [])


def is_mounted(path: Path) -> bool:
    """True if *path* is a mount point (of any kind)."""
    return os.path.ismount(str(path))


def _mount_pairs(cfg_local, cfg_remote) -> list[tuple[Path, Path]]:
    """(remote_dir, local_dir) pairs to mount, deduplicated and in order."""
    pairs: list[tuple[Path, Path]] = []
    for attr in ("path_raw", "path_preprocess"):
        remote, local = getattr(cfg_remote, attr, None), getattr(cfg_local, attr, None)
        if remote is None or local is None:
            continue
        pair = (Path(remote), Path(local))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def mount(cfg_local, cfg_remote, host: str, *, unmount: bool = False,
          dry_run: bool = False, read_only: bool = False) -> None:
    """sshfs-mount (or unmount) the dataset's directories from *host*."""
    pairs = _mount_pairs(cfg_local, cfg_remote)
    if not pairs:
        print("Dataset config defines neither path_raw nor path_preprocess — nothing to mount.",
              file=sys.stderr)
        sys.exit(1)

    tool = "fusermount" if unmount else "sshfs"
    if shutil.which(tool) is None and not dry_run:
        hint = ("fusermount comes with the fuse package" if unmount
                else "install it with e.g. 'sudo apt install sshfs'")
        print(f"'{tool}' not found — {hint}.", file=sys.stderr)
        sys.exit(1)

    failed = 0
    for remote, local in pairs:
        if unmount:
            if not is_mounted(local):
                print(f"  not mounted: {local}")
                continue
            cmd = ["fusermount", "-u", str(local)]
        else:
            if is_mounted(local):
                # A live mount keeps the options it was made with, so one from
                # before max_conns is still serialising every request.
                print(f"  already mounted: {local}"
                      "  (--unmount and mount again to pick up option changes)")
                continue
            if local.exists() and any(local.iterdir()):
                # Mounting over a non-empty directory hides whatever is in it —
                # exactly the local-leftovers collision this command exists to
                # avoid, so refuse rather than paper over it.
                print(
                    f"ERROR: {local} exists and is not empty; refusing to mount over it.\n"
                    f"       Move it aside, or point the config's path at an unused directory.",
                    file=sys.stderr,
                )
                failed += 1
                continue
            if not dry_run:
                local.mkdir(parents=True, exist_ok=True)
            options = sshfs_options(read_only)
            cmd = ["sshfs", f"{host}:{remote}", str(local), "-o", ",".join(options)]

        print("  " + " ".join(cmd))
        if dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed += 1
            print(f"  FAILED ({result.returncode}): {' '.join(cmd)}", file=sys.stderr)

    if failed:
        sys.exit(1)
    if not dry_run and not unmount:
        print("\nMounted. Unmount again with the same command plus --unmount.")
