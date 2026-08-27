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
which mirrors ``sharded_name`` and so interpolates ``${category}``: the repo
holds one subset per split-variant and category, named exactly like the cache
directory it came from.

    GenIntelLab/HouseCorr3D/
      README.md                    ← configs: block, one entry per subset
      hc3d_frame_object_train_n50000_b100_r512_mmc16_cbackpack/
        train-00000-of-000NN.parquet …
      hc3d_frame_object_train_n50000_b100_r512_mmc16_cbackpack_meshes/
        train-00000-of-00001.parquet          ← mesh sidecar
      hc3d_frame_object_train_n50000_b100_r512_mmc16_cbook/ …
      hc3d_frame_object_test_n1000_b100_r512_mmc16_cbackpack/ …
      hc3d_frame_object_test_real_n1000_b100_r512_mmc16_cbread/ …
      …

Uploading is one command per category (``-a "-c backpack,-c book"`` runs
several, each pushing its own subset into that repo) and a consumer pulls only
the subset it needs.  Carrying the shard parameters (item cap, shard size,
resolution, mesh type) in the subset name lets caches that differ only in those
sit side by side rather than overwriting one another; re-pushing a subset
replaces exactly its own shards.  ``huggingface_config_name`` exists as a
separate field because it is a *hub* name: it defaults to ``sharded_name`` but
can be set to anything, and the local cache directory is always found under
``sharded_name`` regardless.

Because the subsets share one repo, per-category uploads that run concurrently
(``-a`` with ``--remote`` submits one sbatch job each) all rewrite the same
README, and two things bite that ``upload_sharded`` has to handle itself:

* the card's ``dataset_info:`` block grows without bound and the Hub rejects
  the front matter with 413 past ~1 MB — see ``_strip_dataset_info``;
* the card commit carries a ``parent_commit`` precondition, so the losers of the
  race get 412.  ``datasets`` looks like it retries that, but the check is dead
  code against ``huggingface_hub`` 1.x — see ``_push_with_retry``.

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

    ``huggingface_config_name`` when the config sets one — the hc3d configs set
    it to ``${sharded_name}``, so their subsets are named after the cache
    directory even though they all share one repo.  Otherwise the cache
    directory name itself, which is the same thing by default; the field exists
    so a config can publish under a name of its own without moving the cache.
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

        # Only before the push, and only once the card is actually large — see
        # _strip_dataset_info on why each extra commit here is expensive.
        _strip_dataset_info(api, repo_id)
        _push_with_retry(
            load_from_disk(str(path)), api, repo_id,
            config_name=config_name, split=split,
            private=private, token=token,
            commit_message=f"{'update' if config_name in uploaded else 'add'} {config_name}",
        )
        # The mesh sidecar is a dataset of its own (object_id → mesh), stored
        # next to the shards locally; on the hub it is a second config so the
        # item shards stay one clean table for the viewer.
        side = path / "meshes"
        if side.is_dir():
            _push_with_retry(
                load_from_disk(str(side)), api, repo_id,
                config_name=_mesh_config(config_name), split=split,
                private=private, token=token,
                commit_message=f"meshes for {config_name}",
            )
    except Exception as e:
        _reraise_with_identity(e, api, platform, repo_id)
    print(f"Done. https://huggingface.co/datasets/{repo_id}/viewer/{config_name}")


def _is_commit_conflict(exc: Exception) -> bool:
    """True for the Hub's "someone else committed first" responses (409 / 412).

    Reads the status off the response, never out of the message — see
    ``_http_status`` for why grepping the text misfires.
    """
    return _http_status(exc) in (409, 412)


def _push_with_retry(dset, api, repo_id: str, *, attempts: int = 10, **kwargs) -> None:
    """``dset.push_to_hub(repo_id, **kwargs)``, retried on a commit conflict.

    ``datasets`` has its own retry for this, but it is dead code against
    ``huggingface_hub`` 1.x: it re-raises unless ``err.__context__`` is itself
    an ``HfHubHTTPError``, and the hub now raises ``HfHubHTTPError`` *from* an
    ``httpx.HTTPStatusError``, so the isinstance check never matches.  With one
    shared repo and ~50 concurrent per-category uploads, the losers of that race
    simply died with 412 — the Parquet already pushed, only the card commit lost.

    Retrying re-reads the card and rebuilds the commit; the already-uploaded
    Parquet is deduplicated by the Hub, so a retry is cheap.
    """
    import random
    import time

    for attempt in range(attempts):
        try:
            dset.push_to_hub(repo_id, **kwargs)
            return
        except Exception as e:
            if attempt == attempts - 1 or not _is_commit_conflict(e):
                raise
            delay = min(2 ** attempt, 30) * (1 + random.random())
            print(f"  commit conflict on {repo_id} — retrying in {delay:.0f}s "
                  f"({attempt + 1}/{attempts - 1})")
            time.sleep(delay)
            _strip_dataset_info(api, repo_id)


def _strip_dataset_info(api, repo_id: str, attempts: int = 8,
                        min_bytes: int = 300_000) -> None:
    """Drop the card's ``dataset_info:`` block, keeping ``configs:``.

    Only acts once the card has grown past *min_bytes*, because **every commit
    counts against the Hub's rate limit of 128 repository commits per hour** and
    a bulk upload is already spending ~2 of them per category.  Stripping on
    every push cost a third (and a fourth, for the mesh sidecar), which is what
    exhausted the quota mid-run and 429'd the remaining jobs.  At ~38 kB per
    block the default threshold leaves ~18 blocks of headroom under the ~1 MB
    ceiling while firing roughly once per eight uploads.

    ``push_to_hub`` embeds the full Arrow feature schema of every config in the
    card's YAML front matter — ~38 kB per subset for these nested tensor
    records, against ~85 bytes for the ``configs:`` entry the dataset viewer
    actually needs.  The Hub validates that front matter on every commit
    (``/api/validate-yaml``) and rejects it with **413 Payload Too Large** past
    ~1 MB, which one shared repo reaches at ~26 subsets: every upload after that
    fails, however small its own data is.

    The block is rebuilt from whatever the card already carries (see
    ``datasets._get_updated_dataset_card``), so removing it after each push
    bounds the card at ``configs:`` plus the block the next push adds for
    itself.  Nothing reads it back — ``load_from_hub`` goes through
    ``get_dataset_config_names`` / ``load_dataset``, both of which work off
    ``configs:`` and the Parquet's own schema.  The cost is the row counts and
    byte sizes the Hub shows on the card.

    Concurrent uploads race here; the commit carries a ``parent_commit``
    precondition and is retried, and losing the race is harmless anyway — the
    next push strips whatever is left.
    """
    import yaml
    from huggingface_hub import CommitOperationAdd, HfFileSystem
    from huggingface_hub.errors import HfHubHTTPError

    fs = HfFileSystem(token=api.token)
    for attempt in range(attempts):
        try:
            parent = api.repo_info(repo_id, repo_type="dataset").sha
            text = fs.read_text(f"datasets/{repo_id}/README.md", revision=parent)
        except Exception:
            return  # no card yet (or unreadable) — nothing to strip
        if not text.startswith("---"):
            return
        if len(text) < min_bytes:
            return  # still small — not worth a commit against the hourly quota
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
        if "dataset_info" not in meta:
            return
        meta.pop("dataset_info")
        new = "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---" + body
        try:
            api.create_commit(
                repo_id=repo_id, repo_type="dataset",
                operations=[CommitOperationAdd("README.md", new.encode())],
                commit_message="strip dataset_info from card (keeps YAML under the Hub's 1 MB limit)",
                parent_commit=parent,
            )
            return
        except HfHubHTTPError as e:
            if attempt == attempts - 1:
                print(f"WARNING: could not strip dataset_info from the card: {e}")
                return
            import random, time
            time.sleep((1 + attempt) * (1 + random.random()))


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


def _http_status(exc: Exception) -> "int | None":
    """The HTTP status behind *exc*, from the response rather than its text.

    The message is not safe to grep: every Hub error carries a request id like
    ``Root=1-6a9027a9-74851641184448c403c122b9``, and a plain ``"403" in text``
    matches inside it — which turned a 429 rate-limit into a bogus "cannot write
    to this namespace" report.
    """
    seen, e = set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status is not None:
            return int(status)
        e = e.__context__ or e.__cause__
    return None


def _reraise_with_identity(exc: Exception, api, platform: str, repo_id: str):
    """Add who the token belongs to to a permission error, then re-raise."""
    text = str(exc)
    status = _http_status(exc)
    if status == 429 or "rate limit" in text.lower():
        raise RuntimeError(
            f"{text}\n"
            f"  the Hub allows 128 repository commits per hour and this bulk upload "
            f"exhausted them. Each category costs ~2 commits (its shards, and the "
            f"mesh sidecar), so ~50 categories per hour is the ceiling — split the "
            f"-a ablation list into waves and leave an hour between them, or wait "
            f"out the window and re-run with --override (finished categories are "
            f"skipped without it)."
        ) from exc
    if status in (401, 403) or "Forbidden" in text or "Unauthorized" in text:
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
