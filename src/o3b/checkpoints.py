"""Move the checkpoints a bench run loads between platforms and the Hub.

    o3b bench rsync     -p slurm_lmbl40_morpheus -b hc3d_crsp3d_frame_object_pair_nn \
        -a method/frame_object/morpheus.yaml,ckpts/morpheus/morpheus.yaml,category/housecorr3d
    o3b bench hf-upload -p slurm_lmbl40_morpheus -b hc3d_crsp3d_frame_object_pair_nn \
        -a method/frame_object/morpheus.yaml,ckpts/morpheus/morpheus.yaml,category/housecorr3d

WHICH CHECKPOINT
----------------
Neither command takes a path.  Both resolve the *same* config a ``bench run``
with those exact ``-b``/``-a``/``-p`` arguments would see (``cli._bench_runs``)
and read ``method.checkpoints_cats`` out of it, restricted to the categories
that run actually evaluates (``method.categories``).  So the file that gets
copied or published is by construction the file the run would load, and the
three commands stay in step when a ``ckpts/`` ablation is re-pointed at a new
training sweep — there is no second place to update.

Since ``checkpoints_cats`` resolves ``${platform.path_exps}``, the same run
names name different absolute paths on different platforms.  That is exactly
what ``rsync`` exploits: it resolves the run twice, once for the source
platform (``-p``) and once for the target (``-t``, the local ``default``
platform unless given), and copies each category's checkpoint from the one to
the other.  The copy therefore lands where a local ``bench run`` with the
unchanged ablation looks for it.

    -p slurm_lmbl40_morpheus  /work/dlclarge1/sommerl-od3d/exps/<run>/morpheus.ckpt
    -t default                /data/lmbraid19/sommerl/workspace/exps/<run>/morpheus.ckpt

HUB LAYOUT
----------
``hf-upload`` publishes into the **model** repo named by
``checkpoints_huggingface.repo_id`` (``GenIntelLab/HouseCorr3D``), one file per
category:

    GenIntelLab/HouseCorr3D/                    ← model repo
      morpheus/backpack/morpheus.ckpt
      morpheus/book/morpheus.ckpt
      …
      common3d/backpack/nemo.ckpt

Model and dataset repos are separate namespaces on the Hub, so this coexists
with the *dataset* repo ``GenIntelLab/HouseCorr3D`` that ``o3b dataset
hf-upload`` writes the sharded caches into (see ``dataset/huggingface.py``) —
same name, different repo type, no collision.  The ``<prefix>/<category>/``
layout keeps the path free of run names and timestamps, so re-uploading a
retrained category replaces its file in place and a consumer can hardcode the
path.

Uploading needs the file on *this* machine, so a ``-p`` that is a remote
platform is fetched first, by the same code path ``rsync`` uses (already-copied
checkpoints are skipped) — running the two commands in sequence therefore
transfers nothing twice.

USING A PUBLISHED CHECKPOINT
----------------------------
``resolve_checkpoint`` accepts an ``hf://<repo_id>/<path in repo>`` URL wherever
a checkpoint path is expected and returns the downloaded local file (cached by
``huggingface_hub``, so a second run does not re-download).  The ``ckpts/``
ablations ending in ``_hf`` point ``checkpoints_cats`` at those URLs, which is
all it takes to evaluate without the cluster:

    o3b bench run -b hc3d_crsp3d_frame_object_pair_nn \
        -a method/frame_object/morpheus.yaml,ckpts/morpheus/morpheus_hf.yaml,category/housecorr3d

CREDENTIALS
-----------
The token comes from ``credentials.huggingface.token`` in the platform config,
exactly as for dataset uploads — ``dataset/huggingface.py:hf_token`` is reused,
including its "print who the token belongs to" reporting.  A public repo needs
no token to download from.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

# Scheme marking a checkpoint that lives on the Hub rather than on a filesystem.
# `hf://<repo_id>/<path>` addresses a *model* repo, matching HfFileSystem, whose
# dataset repos are spelled `hf://datasets/<repo_id>/<path>`.
_HF_SCHEME = "hf://"

# Repo the checkpoints are published to when a config names none.
_DEFAULT_REPO_ID = "GenIntelLab/HouseCorr3D"

# checkpoints_glob patterns allowed to reach a remote shell unquoted.
_SAFE_GLOB = re.compile(r"^[A-Za-z0-9._*?\[\]-]+$")


# ── which checkpoints a run loads ─────────────────────────────────────────────

def run_checkpoints(run_raw: dict) -> dict[str, str]:
    """``{category: checkpoint path}`` for the categories this run evaluates.

    ``method.checkpoints_cats`` lists every category the ``ckpts/`` ablation
    knows about (all 50), but a run only ever loads the ones in
    ``method.categories`` — the method itself skips the rest (see
    ``MorpheusMethod._load_checkpoints``).  Copying or publishing the other 49
    would move gigabytes that this invocation has nothing to do with, and would
    fail on any category the sweep never trained.
    """
    method = run_raw.get("method") or {}
    ckpts = method.get("checkpoints_cats") or {}
    if not isinstance(ckpts, dict):
        return {}
    cats = method.get("categories")
    if isinstance(cats, list) and cats:
        ckpts = {c: ckpts[c] for c in cats if c in ckpts}
    return {c: str(p) for c, p in ckpts.items() if p and str(p) != "None"}


def checkpoint_glob(run_raw: dict) -> str | None:
    """``checkpoints_glob``: the checkpoint is sibling files, not the named one.

    ``checkpoints_cats`` values are a path *inside* the run directory by
    repo-wide convention, and for most methods that path is the checkpoint.
    MagicPony is the exception — ``MagicPonyMethod._get_trainer`` only takes the
    parent directory and hands it to magicpony's own loader, which globs
    ``*.pth`` inside it, so the named ``nemo.ckpt`` need not (and does not)
    exist.  Setting ``checkpoints_glob: "*.pth"`` in that ablation makes these
    commands move the files that are really the checkpoint instead of failing on
    a path nothing ever wrote.
    """
    glob = run_raw.get("checkpoints_glob")
    return str(glob) if glob else None


def _expand_glob(host: str | None, entries: list, glob: str) -> list:
    """Replace each (…, src, dst) by one entry per file matching *glob* next to it."""
    if not _SAFE_GLOB.match(glob):
        raise ValueError(f"refusing to expand unsafe checkpoints_glob: {glob!r}")

    out = []
    for stem, cat, src, dst in entries:
        src_dir, dst_dir = Path(src).parent, Path(dst).parent
        # the pattern must stay unquoted for the remote shell to expand it; it
        # was whitelisted above. nullglob keeps an unmatched pattern from
        # echoing itself back as a literal filename.
        listing = _capture(
            host,
            f"shopt -s nullglob; for f in {shlex.quote(str(src_dir))}/{glob}; do "
            f'[ -f "$f" ] && basename "$f"; done',
        )
        names = sorted(listing.split())
        if not names:
            raise FileNotFoundError(
                f"no file matching {glob!r} in {host or 'local'}:{src_dir}\n"
                f"  (category '{cat}', ablation {stem}). checkpoints_glob says the "
                f"checkpoint is those files rather than {Path(src).name}."
            )
        out += [(stem, cat, str(src_dir / n), str(dst_dir / n)) for n in names]
    return out


def _hub_target(run_raw: dict, repo_id: str | None, prefix: str | None) -> tuple[str, str]:
    """(repo id, path prefix) to publish this run's checkpoints under.

    From ``checkpoints_huggingface:`` in the ``ckpts/`` ablation, overridable
    per invocation by ``--repo`` / ``--prefix``.  The prefix is what keeps two
    methods' checkpoints apart inside the one shared repo, so there is no
    sensible default for it — a missing one is an error rather than a guess
    that would silently publish morpheus weights next to common3d's.
    """
    block = run_raw.get("checkpoints_huggingface") or {}
    repo = repo_id or block.get("repo_id") or _DEFAULT_REPO_ID
    pre = prefix or block.get("prefix")
    if not pre:
        raise ValueError(
            "no hub prefix for these checkpoints — the ablation has no "
            "`checkpoints_huggingface.prefix` and --prefix was not given. Add e.g.\n"
            "  checkpoints_huggingface:\n"
            f"    repo_id: {repo}\n"
            "    prefix: morpheus\n"
            "to the ckpts/ ablation YAML; it names the folder inside the repo that "
            "keeps this method's checkpoints apart from the others."
        )
    return str(repo), str(pre).strip("/")


def hub_path(prefix: str, category: str, local_path: str) -> str:
    """Path inside the repo a category's checkpoint is published at."""
    return f"{prefix}/{category}/{Path(local_path).name}"


def _collect(benchmark: Path, ablation, platform: str | None, target: str):
    """[(combo_stem, category, source path, target path)] for every run.

    The two paths are the same checkpoint as the *source* and the *target*
    platform each name it; pairing them per (run, category) is what makes the
    copy land where a run on the target platform would look.
    """
    from o3b.cli import _bench_runs

    src_platform, _, src_runs = _bench_runs(benchmark, ablation, platform)
    _, _, dst_runs = _bench_runs(benchmark, ablation, target)

    out = []
    seen = set()
    glob = None
    for (_c, src_raw, stem), (_c2, dst_raw, _s2) in zip(src_runs, dst_runs):
        glob = glob or checkpoint_glob(src_raw)
        src_ckpts = run_checkpoints(src_raw)
        dst_ckpts = run_checkpoints(dst_raw)
        for cat, src in src_ckpts.items():
            dst = dst_ckpts.get(cat)
            if dst is None or (src, dst) in seen:
                continue
            seen.add((src, dst))
            out.append((stem, cat, src, dst))

    if glob and out:
        from o3b.dataset.sync import _preflight, _ssh_host

        host = _ssh_host(src_platform)
        if host:
            _preflight(host)  # the listing below runs in BatchMode and cannot prompt
        out = _expand_glob(host, out, glob)
    return src_platform, out


def _require_checkpoints(entries: list) -> None:
    if entries:
        return
    raise ValueError(
        "none of these runs load a checkpoint — `method.checkpoints_cats` is "
        "empty or holds nothing for the evaluated categories. Add a ckpts/ "
        "ablation to -a (e.g. ckpts/morpheus/morpheus.yaml), which is what maps "
        "each category to the run that trained it."
    )


# ── remote/local file primitives ──────────────────────────────────────────────

def _remote_size(host: str | None, path: str) -> int | None:
    """Size of *path* in bytes, or None when it does not exist."""
    if host is None:
        p = Path(path)
        return p.stat().st_size if p.is_file() else None
    out = subprocess.run(
        ["ssh", "-q", "-o", "BatchMode=yes", host,
         f'stat -c %s {shlex.quote(path)} 2>/dev/null || echo __MISSING__'],
        capture_output=True, text=True,
    ).stdout.strip()
    if not out or out == "__MISSING__":
        return None
    try:
        return int(out.splitlines()[-1])
    except ValueError:
        return None


def _copy_file(src_host: str | None, src: str, dst_host: str | None, dst: str) -> None:
    """Copy one file between platforms, creating the target directory."""
    if src_host and dst_host:
        # rsync refuses remote→remote and neither cluster can reach the other,
        # so relay through this machine — the same pipe dataset/sync.py uses.
        from o3b.dataset.sync import _tar_relay
        name = Path(src).name
        # _tar_relay's receiving side does its own `mkdir -p`
        _tar_relay(src_host, Path(src).parent, dst_host, Path(dst).parent, [name],
                   compress=False)
        if Path(src).name != Path(dst).name:
            _run(dst_host, f"mv {shlex.quote(str(Path(dst).parent / name))} {shlex.quote(dst)}")
        return

    _mkdir(dst_host, str(Path(dst).parent))
    src_arg = f"{src_host}:{src}" if src_host else src
    dst_arg = f"{dst_host}:{dst}" if dst_host else dst
    argv = ["rsync", "-a", "--partial", "--info=progress2", src_arg, dst_arg]
    print("  " + " ".join(shlex.quote(a) for a in argv))
    if subprocess.run(argv).returncode != 0:
        raise RuntimeError(f"rsync failed: {src_arg} → {dst_arg}")


def _capture(host: str | None, command: str) -> str:
    """Run a shell command here or on *host*, returning stdout."""
    argv = (["ssh", "-q", "-o", "BatchMode=yes", host, command] if host
            else ["bash", "-c", command])
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed on {host or 'localhost'}: {command}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def _mkdir(host: str | None, directory: str) -> None:
    _run(host, f"mkdir -p {shlex.quote(directory)}")


def _run(host: str | None, command: str) -> None:
    argv = (["ssh", "-q", "-o", "BatchMode=yes", host, command] if host
            else ["bash", "-c", command])
    if subprocess.run(argv).returncode != 0:
        raise RuntimeError(f"command failed on {host or 'localhost'}: {command}")


def _human(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


# ── o3b bench rsync ───────────────────────────────────────────────────────────

def sync_checkpoints(benchmark: Path, ablation, *, platform: str | None = None,
                     target: str = "default", override: bool = False,
                     dry_run: bool = False) -> list[tuple[str, str]]:
    """Copy each run's checkpoint from *platform* to *target*.

    Returns ``[(category, target path)]`` for every checkpoint that is present
    on the target afterwards — what ``hf-upload`` then reads.  A target file of
    the same size as the source is left alone (checkpoints are written once and
    never appended to), so re-running is cheap and ``--override`` is only needed
    after retraining under the same run name.
    """
    from o3b.dataset.sync import _preflight, _ssh_host

    src_platform, entries = _collect(benchmark, ablation, platform, target)
    _require_checkpoints(entries)

    src_host = _ssh_host(src_platform)
    dst_host = _ssh_host(target)
    if src_host == dst_host and src_platform == target:
        raise ValueError(
            f"source and target platform are both '{target}' — nothing to sync. "
            f"Pass the platform the checkpoints were trained on as -p."
        )
    if not dry_run:
        for host in (src_host, dst_host):
            if host:
                _preflight(host)

    print(f"Syncing {len(entries)} checkpoint(s): {src_platform} → {target}")
    done: list[tuple[str, str]] = []
    n_copied = n_skipped = 0
    width = len(str(len(entries)))
    for i, (stem, cat, src, dst) in enumerate(entries, 1):
        prefix = f"[{i:{width}}/{len(entries)}]"
        src_size = _remote_size(src_host, src)
        if src_size is None:
            raise FileNotFoundError(
                f"{prefix} checkpoint missing on {src_platform}: {src}\n"
                f"  (category '{cat}', ablation {stem}). The path is built from the "
                f"ckpts/ ablation's run name — check that the run that trained this "
                f"category is named there and that -p names the platform it ran on."
            )
        dst_size = _remote_size(dst_host, dst)
        if dst_size == src_size and not override:
            n_skipped += 1
            done.append((cat, dst))
            print(f"{prefix} skip   {cat} ({_human(src_size)}, already at {dst})")
            continue

        print(f"{prefix} copy   {cat} ({_human(src_size)})")
        print(f"          {src_host or 'local'}:{src}")
        print(f"       →  {dst_host or 'local'}:{dst}")
        if dry_run:
            n_copied += 1
            continue
        _copy_file(src_host, src, dst_host, dst)
        n_copied += 1
        done.append((cat, dst))

    verb = "would copy" if dry_run else "copied"
    print(f"\n{verb} {n_copied}, skipped {n_skipped} (already present) "
          f"of {len(entries)} checkpoint(s).")
    return done


# ── o3b bench hf-upload ───────────────────────────────────────────────────────

def upload_checkpoints(benchmark: Path, ablation, *, platform: str | None = None,
                       target: str = "default", repo_id: str | None = None,
                       prefix: str | None = None, private: bool = False,
                       override: bool = False, dry_run: bool = False) -> None:
    """Publish each run's checkpoint to its ``checkpoints_huggingface`` repo.

    A checkpoint that is only on a remote platform is fetched to *target* first
    (``sync_checkpoints``), because the upload reads a local file; that step is
    a no-op once ``o3b bench rsync`` has run with the same arguments.
    """
    from o3b.dataset.huggingface import _identity, _reraise_with_identity, hf_token

    src_platform, entries = _collect(benchmark, ablation, platform, target)
    _require_checkpoints(entries)

    _, runs = _bench_runs_for(benchmark, ablation, platform)
    repo, pre = _hub_target(runs[0][1], repo_id, prefix)

    # the upload reads a local file, so anything still on the cluster comes
    # over first — skipped per file when rsync already brought it
    if src_platform != target:
        print(f"Fetching checkpoints from {src_platform} to {target} first…\n")
        sync_checkpoints(benchmark, ablation, platform=platform, target=target,
                         override=override, dry_run=dry_run)
        print()

    uploads = []
    for stem, cat, _src, dst in entries:
        if not dry_run and not Path(dst).is_file():
            raise FileNotFoundError(
                f"checkpoint for '{cat}' not on this machine: {dst}\n"
                f"  (ablation {stem}). Fetch it first with `o3b bench rsync` using "
                f"the same -b/-a/-p arguments."
            )
        uploads.append((cat, dst, hub_path(pre, cat, dst)))

    local = [d for _c, d, _r in uploads if Path(d).is_file()]
    total = sum(Path(d).stat().st_size for d in local)
    print(f"Uploading {len(uploads)} checkpoint(s), {_human(total)} → "
          f"https://huggingface.co/{repo}"
          + (f" ({len(uploads) - len(local)} not fetched yet, so not counted)"
             if len(local) != len(uploads) else ""))

    from huggingface_hub import HfApi
    token = hf_token(target)
    api = HfApi(token=token)
    print(f"        as {_identity(api, target)}\n")

    if dry_run:
        for cat, dst, rel in uploads:
            print(f"  would upload {cat}: {dst} → {repo}/{rel}")
        print("\nDry run — nothing uploaded.")
        return

    n_uploaded = n_skipped = 0
    try:
        api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
        existing = set(api.list_repo_files(repo_id=repo, repo_type="model"))
        width = len(str(len(uploads)))
        for i, (cat, dst, rel) in enumerate(uploads, 1):
            head = f"[{i:{width}}/{len(uploads)}]"
            if rel in existing and not override:
                n_skipped += 1
                print(f"{head} skip   {cat} — {repo}/{rel} exists (--override to replace)")
                continue
            print(f"{head} upload {cat} ({_human(Path(dst).stat().st_size)}) → {rel}")
            api.upload_file(
                path_or_fileobj=dst, path_in_repo=rel, repo_id=repo,
                repo_type="model",
                commit_message=f"{'update' if rel in existing else 'add'} {rel}",
            )
            n_uploaded += 1
    except Exception as e:
        _reraise_with_identity(e, api, target, repo)

    print(f"\nUploaded {n_uploaded}, skipped {n_skipped} of {len(uploads)}. "
          f"https://huggingface.co/{repo}/tree/main/{pre}")


def _bench_runs_for(benchmark: Path, ablation, platform: str | None):
    from o3b.cli import _bench_runs

    plat, _, runs = _bench_runs(benchmark, ablation, platform)
    if not runs:
        raise ValueError(f"no YAML files found in {ablation!r}")
    return plat, runs


# ── using a published checkpoint (hf:// paths in checkpoints_cats) ────────────

def is_hf_checkpoint(path) -> bool:
    return str(path).startswith(_HF_SCHEME)


def resolve_checkpoint(path, platform: str = "default") -> str:
    """Local file for *path*, downloading it when it is an ``hf://`` URL.

    Plain filesystem paths are returned untouched, so a method can call this on
    every ``checkpoints_cats`` value regardless of where it points.  The
    download goes through ``hf_hub_download``, which caches under ``HF_HOME`` /
    ``HF_HUB_CACHE`` — a second run of the same ablation hits the cache.
    """
    if not is_hf_checkpoint(path):
        return str(path)

    from huggingface_hub import hf_hub_download

    from o3b.dataset.huggingface import hf_token

    rest = str(path)[len(_HF_SCHEME):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"malformed checkpoint URL {path!r} — expected "
            f"hf://<owner>/<repo>/<path inside the repo>"
        )
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    return hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="model",
        token=hf_token(platform) or None,
    )


# ── CLI entry points (see o3b/cli.py) ─────────────────────────────────────────

def run_bench_rsync(args) -> None:
    try:
        sync_checkpoints(
            args.benchmark, args.ablation,
            platform=args.platform, target=args.target,
            override=args.override, dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def run_bench_hf_upload(args) -> None:
    try:
        upload_checkpoints(
            args.benchmark, args.ablation,
            platform=args.platform, target=args.target,
            repo_id=args.repo, prefix=args.prefix,
            private=args.private, override=args.override,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
