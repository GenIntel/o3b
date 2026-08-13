"""Push / pull a materialised sharded dataset cache to the HuggingFace Hub.

    o3b dataset hf-upload -d hc3d_frame_object_test_pair_cat_sharded --override \
        -p slurm_lmbl40 --remote -a "-c backpack"

LAYOUT
------
A sharded cache lives at ``<path_preprocess>/sharded/<sharded_name>`` (see
``ConfigurableDataset._sharded_dir``).  ``sharded_name`` carries the category
(``..._c${category}``), so one dataset config has one directory *per category*.
``huggingface_name`` interpolates ``${category}`` too, which makes it one hub
repo per category, holding the cache in a folder named like the local one:

    <name>_cbackpack/
      hc3d_frame_object_test_n1000_b100_r512_mmc16_cbackpack/
        dataset_info.json, state.json, data-*.arrow, meshes/…
    <name>_cbook/
      hc3d_frame_object_test_n1000_b100_r512_mmc16_cbook/
        …

so uploading is one command per category (``-a "-c backpack,-c book"`` runs
several, each into its own repo) and a consumer clones only the category it
needs.  Keeping the ``sharded_name`` folder inside the repo lets caches that
differ only in shard parameters (item cap, shard size, resolution, mesh type)
sit side by side there instead of overwriting one another.

DOWNLOADING
-----------
``use_huggingface: true`` in the dataset config makes ``_setup_sharded`` call
``download_sharded`` instead of building the cache from raw data.  The folder
lands in exactly the place a local build would have written it, so a machine
that already has the cache never contacts the hub, and everything downstream
(``read_sharded_dataset``, the mesh sidecar) is unchanged.

CREDENTIALS
-----------
The token comes from ``credentials.huggingface.token`` in the platform config
(``configs/platform/credentials/custom/default_custom.yaml``), falling back to
whatever ``huggingface_hub`` finds itself (``HF_TOKEN``, ``huggingface-cli
login``).  With ``--remote`` the token is exported into the job as ``HF_TOKEN``
by ``_run_bench_sbatch_cmd``, since the compute node has no ~/.cache login.
Downloading a public repo needs no token at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Same directory level the local cache uses (see ConfigurableDataset._sharded_dir).
_SHARDED_PARENT = "sharded"


def _resolve_cfg(config_path: Path, platform: str, categories: str | None):
    """Load the dataset config with the platform's paths and -c categories applied."""
    from o3b.cli import _categories_to_dataset_overrides
    from o3b.dataset.cli import _platform_to_dataset_overrides
    from o3b.dataset.dataset import DatasetConfig

    overrides = (_platform_to_dataset_overrides(platform)
                 + _categories_to_dataset_overrides(categories))
    return DatasetConfig.from_yaml(Path(config_path), overrides=overrides)


def _require_hub_target(cfg, what: str) -> tuple[str, str]:
    """(huggingface_name, sharded_name), or a clear error about what is missing."""
    if not cfg.huggingface_name:
        raise ValueError(
            f"the dataset config has no 'huggingface_name' — there is no hub repo "
            f"to {what}. Add e.g. 'huggingface_name: GenIntel/<repo>' to it."
        )
    if not cfg.sharded_name:
        raise ValueError(
            f"the dataset config has no 'sharded_name' — only a sharded cache can "
            f"be {what}ed. Use a *_sharded config."
        )
    return str(cfg.huggingface_name), str(cfg.sharded_name)


def hf_token(platform: str = "default") -> str | None:
    """``credentials.huggingface.token`` for *platform*, or None to let the hub decide."""
    from o3b.cli import _credential, _load_platform_config

    try:
        cfg, _ = _load_platform_config(platform)
    except Exception:
        return None
    return _credential(cfg, "huggingface.token") or None


# ── download (use_huggingface: true) ──────────────────────────────────────────

def download_sharded(cfg, platform: str = "default") -> Path:
    """Download ``<huggingface_name>/<sharded_name>`` into the local shard dir.

    Returns the directory ``read_sharded_dataset`` can be pointed at, i.e. the
    same ``<path_preprocess>/sharded/<sharded_name>`` a local build produces.
    Re-runs are incremental: files already present with a matching hash are not
    fetched again.
    """
    from huggingface_hub import snapshot_download

    repo_id, folder = _require_hub_target(cfg, "download from")
    if cfg.path_preprocess is None:
        raise ValueError(
            "use_huggingface is set but path_preprocess is None; there is nowhere "
            "to download the shards to."
        )
    root = Path(cfg.path_preprocess) / _SHARDED_PARENT
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}/{folder} → {root / folder} …")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{folder}/*"],
        local_dir=str(root),
        token=hf_token(platform),
    )
    path = root / folder
    if not path.exists():
        raise FileNotFoundError(
            f"{repo_id} has no folder '{folder}'. Either the cache for this "
            f"category was never uploaded (o3b dataset hf-upload -d <config> "
            f"-c <category>), or the config's sharded_name has changed since."
        )
    return path


# ── upload (o3b dataset hf-upload) ────────────────────────────────────────────

def _dir_size(path: Path) -> tuple[int, int]:
    """(number of files, total bytes) below *path*."""
    files = [f for f in path.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def upload_sharded(config_path: Path, *, platform: str = "default",
                   categories: str | None = None, override: bool = False,
                   private: bool = True, dry_run: bool = False) -> None:
    """Upload this config's built sharded cache to its ``huggingface_name`` repo.

    The cache directory selected by ``-c/--categories`` (via ``${category}`` in
    ``sharded_name``) becomes the identically-named folder in the repo.  An
    already-uploaded folder is left alone unless ``override`` is given, which
    replaces it (files in the repo that the local directory no longer has are
    deleted, so a rebuilt cache does not leave stale shards behind).
    """
    cfg = _resolve_cfg(Path(config_path), platform, categories)
    repo_id, folder = _require_hub_target(cfg, "upload to")
    if cfg.path_preprocess is None:
        raise ValueError(f"platform '{platform}' resolves no path_preprocess for {config_path}")

    path = Path(cfg.path_preprocess) / _SHARDED_PARENT / folder
    if not path.is_dir():
        raise FileNotFoundError(
            f"no sharded cache at {path} — build it first with\n"
            f"  o3b dataset init -d {Path(config_path).stem} -p {platform}"
            + (f" -c {categories}" if categories else "")
        )

    n_files, n_bytes = _dir_size(path)
    size = (f"{n_bytes / 2**30:.2f} GiB" if n_bytes >= 2**30 else
            f"{n_bytes / 2**20:.1f} MiB")
    print(f"Uploading {path} ({n_files} files, {size})")
    print(f"        → https://huggingface.co/datasets/{repo_id}/tree/main/{folder}")

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token(platform))

    if dry_run:
        print("Dry run — nothing uploaded.")
        return

    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    existing = [f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                if f.startswith(f"{folder}/")]
    if existing and not override:
        print(f"{repo_id} already holds {len(existing)} file(s) under '{folder}/' — "
              f"skipping (pass --override to replace them).")
        return

    api.upload_folder(
        folder_path=str(path),
        path_in_repo=folder,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"{'update' if existing else 'add'} {folder}",
        # drop shards the rebuilt cache no longer contains (only inside `folder`)
        delete_patterns="*" if existing else None,
    )
    print(f"Done. https://huggingface.co/datasets/{repo_id}/tree/main/{folder}")


def run_hf_upload(args) -> None:
    """`o3b dataset hf-upload` entry point (see o3b/cli.py)."""
    try:
        upload_sharded(
            args.config,
            platform=args.platform,
            categories=args.categories,
            override=args.override,
            private=not args.public,
            dry_run=args.dry_run,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
