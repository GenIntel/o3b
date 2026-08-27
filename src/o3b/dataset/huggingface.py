"""Push / pull a materialised sharded dataset cache to the HuggingFace Hub.

    o3b dataset hf-upload -d hc3d_frame_object_test_pair_cat_sharded --override \
        -p slurm_lmbl40 --remote -a "-c backpack"

LAYOUT — AND WHY PARQUET
------------------------
A sharded cache lives at ``<path_preprocess>/sharded/<sharded_name>`` (see
``ConfigurableDataset._sharded_dir``) in ``save_to_disk`` form: ``.arrow`` files
plus ``state.json``.  **That format cannot be served by the Hub.**  Uploading
the directory as-is gives a repo with no dataset viewer and no Croissant
metadata — both are produced from Parquet data files declared in the README,
which ``save_to_disk`` writes nothing of.

So the cache is converted on upload: ``Dataset.push_to_hub`` writes Parquet
shards and maintains the README ``configs:`` block that the viewer reads.  Each
cache becomes one *config* — what the Hub UI calls a **subset** — of the
``huggingface_name`` repo.  All hc3d caches share the single repo
``GenIntelLab/HouseCorr3D`` and are told apart by ``huggingface_config_name``,
which interpolates ``${category}``, so the repo holds one subset per
split-variant and category:

    GenIntelLab/HouseCorr3D/
      README.md                    ← configs: block, one entry per subset
      train_backpack/
        train-00000-of-000NN.parquet …
      train_backpack_meshes/
        train-00000-of-00001.parquet          ← mesh sidecar
      train_book/ …
      test_pair_backpack/ …                   ← hc3d_frame_object_test_pair
      test_real_pair_bread/ …                 ← hc3d_frame_object_test_real_pair
      …

Uploading is one command per category (``-a "-c backpack,-c book"`` runs
several, each pushing its own subset into that repo) and a consumer pulls only
the subset it needs.  ``huggingface_config_name`` deliberately drops the shard
parameters (item cap, shard size, resolution, mesh type) that ``sharded_name``
carries, so a rebuilt cache replaces its subset in place rather than adding a
near-duplicate; a config that wants the old side-by-side behaviour just leaves
``huggingface_config_name`` unset and is published under ``sharded_name``.

Because the subsets share one repo, per-category uploads run concurrently
(``-a`` with ``--remote`` submits one sbatch job each) all rewrite the same
README.  ``datasets.push_to_hub`` guards that with a ``parent_commit``
precondition and retries the card commit on 409/412, re-reading the card each
time, so concurrent pushes serialise instead of losing each other's entries.

The Parquet round-trip is lossless — same Arrow schema, same nested tensor
records — but the viewer renders the tensor blobs as (truncated) binary, since
this codec stores zstd-compressed bytes rather than ``datasets`` Image features.
It gives browsable rows, column/schema documentation and Croissant, not image
previews.

DOWNLOADING
-----------
``use_huggingface: true`` in the dataset config makes ``_setup_sharded`` call
``load_from_hub`` instead of building the cache from raw data.  It is a plain
``load_dataset``, so the Parquet lands in the ``datasets`` cache
(``HF_DATASETS_CACHE``, set per platform by ``path_hf_datasets_cache``) and
everything downstream (record decoding, the mesh sidecar) is unchanged.  A
machine that already has the local ``sharded/<sharded_name>`` directory keeps
using it and never contacts the hub.

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


def hub_config_name(cfg) -> str | None:
    """The subset (hub config) this cache is published as.

    ``huggingface_config_name`` when the config sets one — the hc3d configs do,
    since they all share one repo and need short, stable subset names that do
    not carry the local cache's shard parameters.  Otherwise the cache directory
    name itself, which is what publishing a single-cache repo amounts to.
    """
    return cfg.huggingface_config_name or cfg.sharded_name


def _require_hub_target(cfg, what: str) -> tuple[str, str]:
    """(huggingface_name, hub config name), or a clear error about what is missing."""
    if not cfg.huggingface_name:
        raise ValueError(
            f"the dataset config has no 'huggingface_name' — there is no hub repo "
            f"to {what}. Add e.g. 'huggingface_name: GenIntelLab/HouseCorr3D' to it."
        )
    if not cfg.sharded_name:
        raise ValueError(
            f"the dataset config has no 'sharded_name' — only a sharded cache can "
            f"be {what}ed. Use a *_sharded config."
        )
    return str(cfg.huggingface_name), str(hub_config_name(cfg))


def hf_token(platform: str = "default") -> str | None:
    """``credentials.huggingface.token`` for *platform*, or None to let the hub decide."""
    from o3b.cli import _credential, _load_platform_config

    try:
        cfg, _ = _load_platform_config(platform)
    except Exception:
        return None
    return _credential(cfg, "huggingface.token") or None


# ── download (use_huggingface: true) ──────────────────────────────────────────

def _mesh_config(config_name: str) -> str:
    """Config name holding the deduplicated meshes belonging to *config_name*."""
    return f"{config_name}_meshes"


def load_from_hub(cfg, platform: str = "default"):
    """Load this config's cache from the hub as ``(shards, meshes, mesh_rows)``.

    The triple is what ``_setup_sharded`` keeps: the items, the mesh sidecar
    dataset (or None) and its ``{object_id: row}`` map — the same shapes the
    local ``read_sharded_dataset`` / ``read_mesh_sidecar`` pair returns, so
    nothing downstream can tell the two apart.
    """
    from datasets import get_dataset_config_names, load_dataset

    repo_id, config_name = _require_hub_target(cfg, "download from")
    token = hf_token(platform)

    configs = get_dataset_config_names(repo_id, token=token)
    if config_name not in configs:
        raise FileNotFoundError(
            f"{repo_id} has no subset '{config_name}' (it has: {configs}). Either "
            f"the cache for this category was never uploaded (o3b dataset hf-upload "
            f"-d <config> -c <category>), or huggingface_config_name has changed."
        )

    print(f"Loading {repo_id} (config {config_name}) from the HuggingFace Hub…")
    shards = _load_split(repo_id, config_name, cfg, token)

    meshes = mesh_rows = None
    if _mesh_config(config_name) in configs:
        meshes = _load_split(repo_id, _mesh_config(config_name), cfg, token)
        mesh_rows = {oid: i for i, oid in enumerate(meshes["object_id"])}
    return shards, meshes, mesh_rows


def _load_split(repo_id: str, config_name: str, cfg, token):
    """The one split of a config — ``cfg.split`` when present, else whatever is."""
    from datasets import load_dataset

    dsets = load_dataset(repo_id, name=config_name, token=token)
    if cfg.split in dsets:
        return dsets[cfg.split]
    # uploads name the split after cfg.split, but a repo built with a different
    # config revision may carry another name; with one split there is no ambiguity
    return dsets[next(iter(dsets))]


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
    ``sharded_name``) is converted to Parquet and pushed as the config of that
    name — which is what gives the repo a dataset viewer and Croissant metadata
    (see the module docstring).  Its mesh sidecar becomes a second config.  An
    already-uploaded config is left alone unless ``override`` is given, which
    replaces it; push_to_hub then drops the shards the rebuilt cache no longer
    has, so nothing stale survives.
    """
    cfg = _resolve_cfg(Path(config_path), platform, categories)
    repo_id, config_name = _require_hub_target(cfg, "upload to")
    if cfg.path_preprocess is None:
        raise ValueError(f"platform '{platform}' resolves no path_preprocess for {config_path}")

    # the cache directory is named after sharded_name; the *subset* it is
    # published as is config_name, which the config may rename (see hub_config_name)
    path = Path(cfg.path_preprocess) / _SHARDED_PARENT / cfg.sharded_name
    if not path.is_dir():
        raise FileNotFoundError(
            f"no sharded cache at {path} — build it first with\n"
            f"  o3b dataset init -d {Path(config_path).stem} -p {platform}"
            + (f" -c {categories}" if categories else "")
        )

    n_files, n_bytes = _dir_size(path)
    size = (f"{n_bytes / 2**30:.2f} GiB" if n_bytes >= 2**30 else
            f"{n_bytes / 2**20:.1f} MiB")
    split = cfg.split or "train"
    print(f"Uploading {path} ({n_files} files, {size}) as Parquet")
    print(f"        → https://huggingface.co/datasets/{repo_id}/viewer/{config_name}")

    from huggingface_hub import HfApi
    token = hf_token(platform)
    api = HfApi(token=token)
    print(f"        as {_identity(api, platform)}")

    if dry_run:
        print("Dry run — nothing uploaded.")
        return

    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

        uploaded = _uploaded_configs(api, repo_id)
        if config_name in uploaded and not override:
            print(f"{repo_id} already holds config '{config_name}' — skipping "
                  f"(pass --override to replace it).")
            return

        _delete_non_parquet(api, repo_id, config_name)

        from datasets import load_from_disk

        load_from_disk(str(path)).push_to_hub(
            repo_id, config_name=config_name, split=split,
            private=private, token=token,
            commit_message=f"{'update' if config_name in uploaded else 'add'} {config_name}",
        )
        # The mesh sidecar is a dataset of its own (object_id → mesh), stored
        # next to the shards locally; on the hub it is a second config so the
        # item shards stay one clean table for the viewer.
        side = path / "meshes"
        if side.is_dir():
            load_from_disk(str(side)).push_to_hub(
                repo_id, config_name=_mesh_config(config_name), split=split,
                private=private, token=token,
                commit_message=f"meshes for {config_name}",
            )
    except Exception as e:
        _reraise_with_identity(e, api, platform, repo_id)
    print(f"Done. https://huggingface.co/datasets/{repo_id}/viewer/{config_name}")


def _delete_non_parquet(api, repo_id: str, config_name: str) -> None:
    """Drop anything under ``<config_name>/`` that is not a Parquet shard.

    Earlier versions uploaded the ``save_to_disk`` directory verbatim, so a repo
    written by one of those has ``data-*.arrow`` / ``state.json`` sitting in the
    very folder the Parquet config now uses.  push_to_hub only replaces the
    ``<split>-*`` files it wrote itself, which would leave that dead payload
    behind — several GB of it, in a folder the viewer scans.
    """
    from huggingface_hub import CommitOperationDelete

    prefix = f"{config_name}/"
    try:
        stale = [f for f in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                 if f.startswith(prefix) and not f.endswith(".parquet")]
    except Exception:
        return
    if not stale:
        return
    print(f"Removing {len(stale)} non-Parquet file(s) left in '{config_name}/' "
          f"by an earlier upload…")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
        commit_message=f"drop non-parquet files under {config_name}",
    )


def _uploaded_configs(api, repo_id: str) -> set[str]:
    """Config names already in the repo, read off the data files it holds.

    Cheaper and more robust than parsing the README ``configs:`` block, and it
    also catches a config whose card entry was hand-edited away.
    """
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return set()
    return {f.split("/")[0] for f in files if f.endswith(".parquet") and "/" in f}


def _identity(api, platform: str) -> str:
    """"<user> (token: <role>, from <source>)" for the token the upload will use.

    Printed before every upload because the token in the platform credentials
    silently outranks a `huggingface-cli login` on this machine — the two being
    different accounts is otherwise only visible as a 403 on the namespace.
    """
    source = ("credentials.huggingface.token" if hf_token(platform)
              else "huggingface-cli login")
    try:
        who  = api.whoami()
        role = ((who.get("auth") or {}).get("accessToken") or {}).get("role", "?")
        return f"{who.get('name')} (token: {role}, from {source})"
    except Exception as e:
        return f"<not logged in: {type(e).__name__}> (token from {source})"


def _reraise_with_identity(exc: Exception, api, platform: str, repo_id: str):
    """Add who the token belongs to to a permission error, then re-raise."""
    text = str(exc)
    if "403" in text or "401" in text or "Forbidden" in text or "Unauthorized" in text:
        namespace = repo_id.split("/")[0]
        raise PermissionError(
            f"{text}\n"
            f"  the token used is {_identity(api, platform)}, which cannot write to "
            f"'{namespace}'.\n"
            f"  Set a write token for that account in credentials.huggingface.token "
            f"(configs/platform/credentials/custom/default_custom.yaml) — a "
            f"`huggingface-cli login` on this machine is ignored while that "
            f"credential is set, and only the credential reaches a --remote job."
        ) from exc
    raise exc


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
    except (ValueError, FileNotFoundError, PermissionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
