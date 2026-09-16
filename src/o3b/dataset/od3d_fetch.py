"""Shared ``fetch()`` machinery for the datasets migrated from od3d.

Four datasets (PASCAL3D, ImageNet3D, HANDAL, Objectron) were preprocessed by
od3d before o3b existed, and their raw downloads range from a plain zip to a
Google Drive folder to a Google Cloud bucket that no longer serves anonymous
callers.  Rather than let the flakiest of them dictate the interface, every one
of them goes through :func:`fetch_or_copy`, which tries the download and falls
back to copying an existing tree.

The fallback is not a convenience — it is the normal path on the cluster, where
all four datasets already exist under both ``/work/dlclarge1/sommerl-od3d/datasets``
and ``/data/lmbraid19/sommerl/datasets``.  A fetch there should cost nothing and
must not fail because Drive rate-limited us or a bucket wants credentials.

Ordering is deliberate:

1. **already present** → report and stop.  Never re-download over a good tree.
2. **copy source** (``fetch_copy_from`` in the config, or a platform default)
   → rsync it in.  Preferred over the network whenever it is reachable, because
   it is both faster and immune to the auth problems below.
3. **download** → only when neither of the above applies.

Step 3 failing is reported as a *skip*, not an exception, whenever a copy source
was configured but incomplete: a half-usable tree the caller can inspect beats a
traceback that hides what was actually found.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence


class FetchSkipped(Exception):
    """Raised when neither download nor copy could provide the data.

    Carries the human-readable reason so the CLI can print instructions (what to
    download by hand, which credentials are needed) instead of a stack trace.
    """


def _is_populated(path: Path, expect: Sequence[str] = ()) -> bool:
    """Does `path` look like a real tree rather than an empty placeholder?

    ``expect`` names subpaths that must exist, which is what distinguishes a
    complete copy from a partial one.  Objectron on the cluster is exactly that
    case: ``Objectron_Preprocess/`` exists and has ``mask/`` and ``depth/``, but
    its ``meta/frames/`` is an empty directory tree, so a bare ``exists()`` test
    would call it done and leave the dataset unusable.
    """
    if not path.is_dir():
        return False
    if not any(path.iterdir()):
        return False
    return all(_has_files(path / sub) for sub in expect)


def _has_files(path: Path) -> bool:
    """True when `path` holds at least one regular file at any depth."""
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    for p in path.rglob("*"):
        if p.is_file():
            return True
    return False


def rsync(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    """Copy `src` into `dst` with rsync, preserving what is already there.

    Trailing slash on the source so the *contents* land in dst rather than a
    nested directory.  ``--ignore-existing`` rather than a plain sync: the point
    is to fill gaps in a tree the cluster already has, never to overwrite files
    another run may be reading.
    """
    src_arg = f"{src}/"
    cmd = ["rsync", "-a", "--info=progress2", "--ignore-existing", src_arg, str(dst)]
    if dry_run:
        print("  (dry-run) " + " ".join(cmd))
        return
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def rsync_files(src_root: Path, dst_root: Path, rel_paths, *, label: str = "",
                dry_run: bool = False) -> int:
    """Copy exactly `rel_paths` (relative to both roots) with one rsync.

    For the case where the source tree is enormously larger than the part that
    is wanted: Objectron's extracted rgb is 910,660 jpgs, of which the benchmark
    samples 2,250.  ``--files-from`` moves that set in a single pass, where a
    blanket directory sync would move hundreds of gigabytes to obtain ~900 MB.

    Returns the number of paths requested (0 when nothing was missing).
    """
    import tempfile

    rel = sorted({str(r) for r in rel_paths})
    if not rel:
        print(f"  {label or 'files'}: nothing to copy")
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".lst", delete=False) as fh:
        fh.write("\n".join(rel) + "\n")
        lst = Path(fh.name)
    try:
        cmd = [
            "rsync", "-a", "--info=progress2", "--ignore-existing",
            f"--files-from={lst}", str(src_root) + "/", str(dst_root),
        ]
        print(f"  {label or 'files'}: {len(rel)} path(s) from {src_root}")
        if dry_run:
            print("  (dry-run) " + " ".join(cmd))
            return len(rel)
        dst_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
    finally:
        lst.unlink(missing_ok=True)
    return len(rel)


def download_zip(url: str, dst: Path, *, strip_top_level: Optional[str] = None) -> None:
    """Download a zip from `url` and extract it into `dst`.

    ``strip_top_level`` names a directory the archive wraps everything in (
    PASCAL3D's zip unpacks to ``PASCAL3D+_release1.1/``); its contents are moved
    up one level afterwards so `dst` holds the tree directly, matching what the
    od3d preprocess paths expect.
    """
    import tempfile
    import urllib.request
    import zipfile

    from o3b.dataset.utils import download_progress

    dst.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path, download_progress)
        print(f"\n  extracting → {dst}")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(dst)
    finally:
        tmp_path.unlink(missing_ok=True)

    if strip_top_level:
        nested = dst / strip_top_level
        if nested.is_dir():
            print(f"  flattening {strip_top_level}/ into {dst}")
            for child in nested.iterdir():
                target = dst / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
            nested.rmdir()


def fetch_or_copy(
    name: str,
    path: Path,
    *,
    expect: Sequence[str] = (),
    copy_from: Optional[Path] = None,
    download: Optional[Callable[[], None]] = None,
    download_hint: str = "",
    dry_run: bool = False,
) -> None:
    """Ensure `path` holds `name`'s data, by presence, copy, or download.

    See the module docstring for why the order is presence → copy → download.
    Raises :class:`FetchSkipped` with `download_hint` when none of the three
    produced a populated tree.
    """
    print(f"[{name}] target: {path}")

    if _is_populated(path, expect):
        print(f"[{name}] already present — nothing to do.")
        return

    if path.is_dir() and any(path.iterdir()):
        missing = [s for s in expect if not _has_files(path / s)]
        print(f"[{name}] present but incomplete (missing: {', '.join(missing)})")

    if copy_from is not None:
        if _is_populated(Path(copy_from), expect):
            print(f"[{name}] copying from {copy_from}")
            rsync(Path(copy_from), path, dry_run=dry_run)
            if dry_run or _is_populated(path, expect):
                print(f"[{name}] done (copied).")
                return
            print(f"[{name}] copy did not complete the tree", file=sys.stderr)
        else:
            print(f"[{name}] copy source unusable or incomplete: {copy_from}")

    if download is not None:
        try:
            download()
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            # A download failure is only fatal when there was no copy source to
            # fall back on; otherwise the caller still has a partial tree worth
            # inspecting, and the hint says what to do by hand.
            print(f"[{name}] download failed: {exc}", file=sys.stderr)
        else:
            if dry_run or _is_populated(path, expect):
                print(f"[{name}] done (downloaded).")
                return

    raise FetchSkipped(
        f"{name}: could not obtain the data at {path}."
        + (f"\n{download_hint}" if download_hint else "")
    )
