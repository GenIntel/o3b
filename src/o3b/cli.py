"""
o3b — o3b command-line interface.

Usage:
  o3b dataset fetch  -d hc3d_object        [--url URL] [--platform PLATFORM]
  o3b dataset index  -d hc3d_object        [--db FILE] [--platform PLATFORM] [--remote]
  o3b dataset init   -d hc3d_object        [--limit N] [--override] [--platform PLATFORM] [--remote]
                                         [-c CATEGORIES] [-a ABLATIONS]
  # -c overrides the config's categories (and ${category}), e.g. -c backpack
  # -a starts one run per comma-separated ablation, each a fragment of extra
  #    init arguments — with --remote one sbatch job per ablation:
  #    o3b dataset init -d <ds> -p <platform> --remote -a "-c backpack,-c book"
  o3b dataset viz    -d hc3d_object_pair   [--db FILE] [--limit N] [--object-id ID]
                                         [--filter-has-kpts] [--render]
                                         [--render-frames N] [--renderer BACKEND]
                                         [--debug] [--platform PLATFORM]
                                         [--port PORT] [--remote]
  # --remote runs the viewer on the platform's compute node (interactive srun)
  # and tunnels its viser port to http://localhost:<--port>.
  # Pair datasets need no separate index step (pairs derived at load time).
  o3b bench run      -b <benchmark> [-p <platform>] [-a <ablation>]
  o3b platform setup    -p <platform>
  o3b platform stop  -p <platform> [-y]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ── dataset sub-parser ────────────────────────────────────────────────────────

def _build_dataset_parser(sub):
    p = sub.add_parser("dataset", help="Dataset commands (fetch, index, init, viz)")
    ds_sub = p.add_subparsers(dest="dataset_command", required=True)

    def _add_config(q):
        from o3b.dataset.cli import _resolve_dataset_config
        q.add_argument(
            "-d", "--config", required=True, type=_resolve_dataset_config, metavar="DATASET",
            help="Dataset config name (e.g. housecorr3d_object_pair, resolved from "
                 "configs/dataset/) or full path to a YAML file",
        )
        q.add_argument(
            "-p", "--platform", default="default", metavar="PLATFORM",
            help="Platform name whose path_datasets_raw / path_datasets_preprocess "
                 "override the dataset config paths (default: default)",
        )
        q.add_argument(
            "-c", "--categories", default=None, metavar="CATEGORIES",
            help="Comma-separated categories overriding the config's 'categories' field "
                 "(e.g. -c backpack or -c backpack,book). Also sets 'category' (joined by "
                 "'_' for several), so ${category} interpolations such as sharded_name follow",
        )

    p_fetch = ds_sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL")

    p_index = ds_sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_index.add_argument("--remove", action="store_true",
                         help="Delete any existing index for this dataset before indexing")
    p_index.add_argument(
        "--max", type=int, default=None, metavar="N", dest="max_index",
        help="Stop after indexing N total rows (for quick testing). "
             "filter_count_max in the config applies at query time only.",
    )
    p_index.add_argument(
        "--remote", action="store_true",
        help="Run this command on the --platform's compute node via `o3b platform run` "
             "instead of indexing locally",
    )

    p_init = ds_sub.add_parser(
        "init",
        help="Instantiate the dataset (builds the sharded cache if configured) without visualising",
    )
    _add_config(p_init)
    p_init.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Additionally load the first N items after construction (default: 0)",
    )
    p_init.add_argument(
        "--override", action="store_true",
        help="Force sharded_override=True — rebuild the sharded cache even if it already exists",
    )
    p_init.add_argument(
        "--remote", action="store_true",
        help="Run this command on the --platform's compute node via `o3b platform run` "
             "instead of initialising locally",
    )
    p_init.add_argument(
        "-a", "--ablation", default=None, metavar="ABLATIONS",
        help="Comma-separated ablations, each a fragment of extra `o3b dataset init` "
             'arguments (e.g. -a "-c backpack,-c book"). One run is started per '
             "ablation — a separate sbatch job each with --remote, otherwise "
             "sequentially in this process",
    )

    p_sync = ds_sub.add_parser(
        "sync-shard",
        help="Copy a built sharded cache from one platform to another "
             "(streamed through this machine — see o3b/dataset/sync.py)",
    )
    from o3b.dataset.cli import _resolve_dataset_config as _resolve_cfg
    p_sync.add_argument(
        "-d", "--config", required=True, type=_resolve_cfg, metavar="DATASET",
        help="Dataset config name (resolved from configs/dataset/) or path to a YAML file; "
             "must define sharded_name",
    )
    p_sync.add_argument(
        "-s", "--source-platform", required=True, metavar="PLATFORM",
        help="Platform to copy the shards from (e.g. slurm)",
    )
    p_sync.add_argument(
        "-t", "--target-platform", required=True, metavar="PLATFORM",
        help="Platform to copy the shards to (e.g. slurm_jupiter)",
    )
    p_sync.add_argument(
        "-c", "--categories", default=None, metavar="CATEGORIES",
        help="Comma-separated categories selecting the shard directory via ${category} "
             "in sharded_name. Use -c '*' to sync every per-category directory",
    )
    p_sync.add_argument(
        "-n", "--dry-run", action="store_true",
        help="List what would be transferred (and the pipeline used) without copying",
    )
    p_sync.add_argument(
        "--override", action="store_true",
        help="Replace a target directory that already exists but differs from the source",
    )
    p_sync.add_argument(
        "--compress", action="store_true",
        help="zstd the stream between the two hosts. Rarely worth it — the shard "
             "payload is already compressed (measured: no speed-up)",
    )
    p_sync.add_argument(
        "-a", "--ablation", default=None, metavar="ABLATIONS",
        help="Comma-separated ablations, each a fragment of extra arguments "
             '(e.g. -a "-c backpack,-c book"); one sync runs per ablation',
    )

    p_vis = ds_sub.add_parser("viz", help="Summarize and optionally render dataset objects")
    _add_config(p_vis)
    p_vis.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_vis.add_argument("--limit", type=int, default=20, metavar="N")
    p_vis.add_argument("--object-id", default=None, metavar="ID")
    p_vis.add_argument(
        "--frame-stride", type=int, default=None, metavar="N",
        help="Initial ←/→ jump size in frames (default: frame_stride from dataset config); "
             "can also be changed via the Stride trackbar",
    )
    p_vis.add_argument(
        "--frames-per-scene", type=int, default=None, metavar="N",
        help="Show a static grid of N evenly-sampled frames per clip instead of the interactive player",
    )
    p_vis.add_argument("--filter-has-kpts", action="store_true")
    p_vis.add_argument("--render", action="store_true")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender")
    p_vis.add_argument("--debug", action="store_true",
                       help="Show front/top/right camera frustums in the viser scene")
    p_vis.add_argument("--object-centric", action="store_true",
                       help="Object-centric view: place object at world origin, camera in object space")
    p_vis.add_argument(
        "--port", type=int, default=None, metavar="PORT",
        help="Pin the viser server to this port (default: viser's own 8080, bumped "
             "if taken). With --remote this is the local port the remote viser "
             "server is tunnelled to",
    )
    p_vis.add_argument(
        "--remote", action="store_true",
        help="Run the visualization on the --platform's compute node via an interactive "
             "srun and tunnel its viser port to localhost:--port",
    )

    p_tform = ds_sub.add_parser(
        "tform",
        help="Interactive axis-convention viewer — determine obj_gl_tform4x4_obj_raw for the dataset",
    )
    _add_config(p_tform)
    p_tform.add_argument("--limit", type=int, default=20, metavar="N",
                         help="Max objects to browse (default: 20)")

    p_pre = ds_sub.add_parser(
        "preprocess",
        help="OpenTT: annotate score bboxes interactively, then extract scores via VLM",
    )
    _add_config(p_pre)
    p_pre.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite output file (default: <path_preprocess>/scoreboards.db)",
    )
    p_pre.add_argument(
        "--annotate", action="store_true",
        help="Draw the scoreboard / left-score / right-score bboxes interactively "
             "for each video (saved to video_bboxes.json). Run this once before VLM.",
    )
    p_pre.add_argument(
        "--model", default="Qwen/Qwen3-VL-2B-Instruct", metavar="MODEL_ID",
        help="HuggingFace model ID for VLM score reading "
             "(default: Qwen/Qwen3-VL-2B-Instruct)",
    )
    p_pre.add_argument(
        "--device", default="cpu", metavar="DEVICE",
        help="Torch device for VLM inference, e.g. cuda:0 (default: cpu)",
    )
    p_pre.add_argument(
        "--video", default=None, metavar="NAME",
        help="Restrict to a single video by name, e.g. game_1 or test_3",
    )
    p_pre.add_argument(
        "--override", action="store_true",
        help="Re-annotate / re-process already-handled videos or frames.",
    )
    p_pre.add_argument(
        "--debug", action="store_true",
        help="Show score crops and raw VLM output during processing.",
    )
    p_pre.add_argument(
        "--remove", action="store_true",
        help="Delete all rows from the scoreboards table and exit.",
    )


def _parse_categories(categories: str | None) -> list[str] | None:
    """Split a comma-separated -c value into a list of category names."""
    if not categories:
        return None
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    return cats or None


def _categories_to_dataset_overrides(categories: str | None) -> list[str]:
    """Return Hydra override strings for the -c / --categories flag.

    ``category`` is set alongside ``categories`` (joined by '_' when several are
    given) so configs interpolating ``${category}`` — e.g. ``sharded_name`` —
    stay in sync with the filtered categories.
    """
    cats = _parse_categories(categories)
    if not cats:
        return []
    return [f"category={'_'.join(cats)}", f"categories=[{', '.join(cats)}]"]


def _dataset_overrides(args) -> list[str]:
    """Platform path overrides plus any -c / --categories override."""
    from o3b.dataset.cli import _platform_to_dataset_overrides
    return (_platform_to_dataset_overrides(args.platform)
            + _categories_to_dataset_overrides(getattr(args, "categories", None)))


def _run_dataset_ablations(args, parser, argv: list[str]) -> None:
    """Start one `o3b dataset <cmd>` run per comma-separated ablation.

    Each ablation is a fragment of extra CLI arguments (e.g. ``-c backpack``);
    it is appended to the original argv and re-parsed, so the ablation's flags
    win over the ones given on the base command line.  With ``--remote`` every
    run is submitted as its own sbatch job, otherwise they run sequentially here.
    """
    import shlex

    ablations = [a.strip() for a in args.ablation.split(",") if a.strip()]
    if not ablations:
        print(f"WARNING: no ablations parsed from {args.ablation!r}", file=sys.stderr)
        return

    print(f"Running {len(ablations)} ablation(s): {ablations}")
    failed: list[str] = []
    for i, ablation in enumerate(ablations, 1):
        print(f"\n── ablation {i}/{len(ablations)}: {ablation} ──")
        sub_args = parser.parse_args(argv + shlex.split(ablation))
        sub_args.ablation = None  # the re-parse carries -a over; drop it to avoid recursion
        try:
            _run_dataset(sub_args)
        except Exception as exc:
            print(f"ERROR: ablation {ablation!r} failed: {exc}", file=sys.stderr)
            failed.append(ablation)

    if failed:
        print(f"\n{len(failed)}/{len(ablations)} ablation(s) failed: {failed}", file=sys.stderr)
        sys.exit(1)


def _run_dataset_remote(args) -> None:
    """Re-invoke `o3b dataset <cmd>` (minus --remote) on the platform's compute node."""
    import shlex

    command = args.dataset_command
    parts = ["o3b", "dataset", command, "-d", args.config.stem, "-p", args.platform]
    if getattr(args, "categories", None):
        parts += ["-c", args.categories]
    if getattr(args, "db", None):
        parts += ["--db", str(args.db)]
    if getattr(args, "remove", False):
        parts.append("--remove")
    if getattr(args, "max_index", None) is not None:
        parts += ["--max", str(args.max_index)]
    if command == "init" and args.limit:
        parts += ["--limit", str(args.limit)]
    if command == "init" and getattr(args, "override", False):
        parts.append("--override")
    remote_cmd = " ".join(shlex.quote(p) for p in parts)

    job_name = f"{command}_{args.config.stem}"
    if cats := _parse_categories(getattr(args, "categories", None)):
        job_name += f"_c{'_'.join(cats)}"  # keep per-ablation jobs distinguishable
    _run_platform_run_cmd(args.platform, remote_cmd, job_name=job_name)


def _viz_remote_command(args, remote_port: int) -> str:
    """Rebuild the `o3b dataset viz` invocation to run on the compute node."""
    import shlex

    parts = [
        "o3b", "dataset", "viz",
        "-d", args.config.stem,
        "-p", args.platform,
        "--port", str(remote_port),
        "--limit", str(args.limit),
    ]
    if args.categories:
        parts += ["-c", args.categories]
    if args.db:
        parts += ["--db", str(args.db)]
    if args.object_id:
        parts += ["--object-id", args.object_id]
    if args.frame_stride is not None:
        parts += ["--frame-stride", str(args.frame_stride)]
    if args.frames_per_scene is not None:
        parts += ["--frames-per-scene", str(args.frames_per_scene)]
    if args.filter_has_kpts:
        parts.append("--filter-has-kpts")
    if args.render:
        parts += ["--render", "--render-frames", str(args.render_frames),
                  "--renderer", args.renderer]
    if args.debug:
        parts.append("--debug")
    if args.object_centric:
        parts.append("--object-centric")
    return " ".join(shlex.quote(p) for p in parts)


def _find_free_local_port(port: int, attempts: int = 20) -> int:
    """Return *port*, or the next free one — ssh -L fails outright on a taken port."""
    import socket

    for candidate in range(port, port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
            except OSError:
                continue
        if candidate != port:
            print(f"Local port {port} is in use — using {candidate} instead.")
        return candidate
    raise RuntimeError(f"No free local port in range {port}–{port + attempts - 1}")


def _run_dataset_viz_remote(args) -> None:
    """Run `o3b dataset viz` on a compute node and tunnel its viser port here.

    Unlike index/init (queued sbatch jobs), the viewer must stay attached: it is
    started with an interactive srun --pty so Ctrl-C ends both the viewer and the
    allocation, while a background thread forwards localhost:<--port> to the
    allocated node once the job is RUNNING.
    """
    import subprocess
    import random
    import uuid

    (ssh_host, srun, repo_path, env_path, path_cuda, path_ws, hf_datasets_cache,
     use_conda, path_conda, mp_env, modules) = _platform_srun_context(args.platform)

    # viser silently increments its port when the requested one is taken, which
    # would leave the tunnel pointing at nothing — a random high port makes a
    # clash on the shared compute node unlikely, and make_viser_server() aborts
    # loudly rather than drifting if it does happen.
    remote_port = random.randint(40000, 60000)
    local_port  = _find_free_local_port(args.port or 8080)
    job_name    = f"o3b_viz_{uuid.uuid4().hex[:8]}"

    init_lines = _srun_env_lines(path_cuda, env_path, repo_path, path_ws, hf_datasets_cache,
                                 use_conda=use_conda, path_conda=path_conda, mp_env=mp_env,
                                 modules=modules)
    init_lines += [
        "",
        f"export O3B_VISER_PORT={remote_port}",
        f'echo "[o3b-viz] viser will listen on $(hostname):{remote_port}"',
        f'echo "[o3b-viz] open http://localhost:{local_port} once it reports it is running"',
        "",
        _viz_remote_command(args, remote_port),
    ]

    remote_init = f"{path_ws}/.od3d_viz_init_{job_name}" if path_ws else f"~/.od3d_viz_init_{job_name}"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_init}"],
        input="\n".join(init_lines), text=True, check=True,
    )

    srun += f" --job-name {job_name}"
    tunnel = _forward_port_once_running(ssh_host, job_name, str(local_port), str(remote_port))
    # the viewer runs from the init file; the shell stays interactive afterwards
    # so the allocation can be reused (e.g. to re-run the viewer) instead of
    # queueing again.
    srun += f" --pty bash --init-file {remote_init}"

    print(f"Starting remote visualization on {ssh_host} (job {job_name})…")
    print(f"Once viser starts, open http://localhost:{local_port}")
    try:
        subprocess.run(["ssh", "-t", ssh_host, srun])
    finally:
        tunnel.stop()
        subprocess.run(["ssh", ssh_host, f"rm -f {remote_init}"], check=False)


def _run_dataset(args, parser=None, argv=None):
    if getattr(args, "ablation", None):
        _run_dataset_ablations(args, parser, argv)
        return
    if args.dataset_command in ("index", "init") and getattr(args, "remote", False):
        _run_dataset_remote(args)
        return
    if args.dataset_command == "sync-shard":
        from o3b.dataset.sync import sync_sharded
        sync_sharded(
            args.config,
            source_platform=args.source_platform,
            target_platform=args.target_platform,
            categories=args.categories,
            override=args.override,
            dry_run=args.dry_run,
            compress=args.compress,
        )
        return
    if args.dataset_command == "viz" and getattr(args, "remote", False):
        _run_dataset_viz_remote(args)
        return
    if args.dataset_command == "viz" and getattr(args, "port", None):
        import os
        os.environ["O3B_VISER_PORT"] = str(args.port)

    from o3b.dataset.cli import _load_class_from_config

    overrides = _dataset_overrides(args)
    cls, cfg = _load_class_from_config(args.config, overrides=overrides)

    if args.dataset_command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.dataset_command == "index":
        cls.index(cfg, db=args.db, remove=args.remove, max_index=getattr(args, "max_index", None))
    elif args.dataset_command == "init":
        cls.init(cfg, limit=args.limit, override=args.override)
    elif args.dataset_command == "viz":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            frame_stride=args.frame_stride,
            frames_per_scene=args.frames_per_scene,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
            debug=args.debug,
            obj_centric=args.object_centric,
        )
    elif args.dataset_command == "tform":
        from o3b.dataset.tform import run_tform_viewer
        run_tform_viewer(cls, cfg, limit=args.limit)
    elif args.dataset_command == "preprocess":
        if not hasattr(cls, "preprocess"):
            print(
                f"ERROR: {cls.__name__} does not implement preprocess().\n"
                "This command is currently only available for OpenTT.",
                file=sys.stderr,
            )
            sys.exit(1)
        cls.preprocess(
            cfg,
            db=args.db,
            model_id=args.model,
            device=args.device,
            video=args.video,
            annotate=args.annotate,
            override=args.override,
            debug=args.debug,
            remove=args.remove,
        )


# ── platform sub-parser ───────────────────────────────────────────────────────

def _build_platform_parser(sub):
    p = sub.add_parser("platform", help="Platform management commands")
    plat_sub = p.add_subparsers(dest="platform_command", required=True)

    p_setup = plat_sub.add_parser(
        "setup",
        help="Copy and run the repository setup script on a remote platform",
    )
    p_setup.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    g_setup = p_setup.add_mutually_exclusive_group()
    g_setup.add_argument(
        "--pull-only", action="store_true",
        help="Only refresh the remote checkout (clone/pull/submodules/credentials) "
             "and stop, skipping the env and dependency install. Forces pull even "
             "where the platform config sets pull: False.",
    )
    g_setup.add_argument(
        "--creds-only", action="store_true",
        help="Only re-install the credentials (the staged credentials/custom/*.yaml "
             "plus the wandb ~/.netrc entry) onto the existing checkout, without "
             "pulling and without the env and dependency install.",
    )

    p_status = plat_sub.add_parser(
        "status",
        help="Show job queue status on a remote platform",
    )
    p_status.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_status.add_argument(
        "--configs", action="store_true",
        help="Also print the resolved platform config",
    )
    p_status.add_argument(
        "--hours", type=float, default=2.0, metavar="N",
        help="Show sacct history for the last N hours (default: 2)",
    )

    p_runi = plat_sub.add_parser(
        "runi",
        help="Open an interactive shell on a compute node via srun --pty bash",
    )
    p_runi.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_runi.add_argument(
        "--forward", default=None, metavar="LOCAL:REMOTE",
        help="Tunnel localhost:LOCAL to the allocated compute node's REMOTE port "
             "(through the platform's ssh host), e.g. --forward 8080:8080",
    )
    p_runi.add_argument(
        "--pull", action="store_true",
        help="Refresh the remote checkout first (`o3b platform setup --pull-only`) so the "
             "shell lands on the pushed HEAD. Needed on setup_on_login platforms such as "
             "JUPITER, whose compute nodes cannot reach github and so never pull by "
             "themselves. Aborts without allocating if the pull fails.",
    )

    p_setupi = plat_sub.add_parser(
        "setupi",
        help="Open an interactive shell, copy setup script, and print it for manual execution",
    )
    p_setupi.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )

    p_run = plat_sub.add_parser(
        "run",
        help="Submit a command as a queued sbatch job on a compute node",
    )
    p_run.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_run.add_argument(
        "-c", "--command", required=True, metavar="CMD",
        help="Shell command to execute on the compute node",
    )

    p_stop = plat_sub.add_parser(
        "stop",
        help="Cancel all running jobs on the platform's configured partition",
    )
    p_stop.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_stop.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt",
    )
    p_stop.add_argument(
        "-j", "--jobs", default=None, metavar="A-B",
        help="Cancel only jobs in the inclusive ID range A-B (e.g. 29175478-29175502)",
    )

    p_queue = plat_sub.add_parser(
        "queue",
        help="Show partition queue with per-job resource usage (GPUs, CPUs, memory)",
    )
    p_queue.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_queue.add_argument(
        "--partition", default=None, metavar="PARTITION",
        help="Override the partition from the platform config",
    )
    p_queue.add_argument(
        "--all-users", action="store_true",
        help="Show jobs from all users (default: only the configured username)",
    )



def _load_platform_config(platform: str):
    """Load a platform config, resolving its defaults chain via OmegaConf merge."""
    from omegaconf import OmegaConf
    from o3b.io import _load_yaml_with_defaults

    configs_dir = (Path(__file__).parent.parent / "configs" / "platform").resolve()
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Platform config directory not found: {configs_dir}")

    cfg_path = configs_dir / f"{platform}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Platform config not found: {cfg_path}")

    cfg = OmegaConf.create(_load_yaml_with_defaults(cfg_path))
    return cfg, configs_dir



def _multiply_metric_with_unit(metric_with_unit: str, factor: int) -> str:
    """Multiply a value-with-unit string (e.g. '5gb') by an integer factor."""
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)(\D+)$", str(metric_with_unit).strip())
    if m:
        return f"{int(float(m.group(1)) * factor)}{m.group(2)}"
    return str(metric_with_unit)


def _save_script_locally(name: str, content: str, ts: str | None = None) -> Path:
    """Save *content* to ~/.o3b/scripts/<timestamp>_<name>.sh and return the path."""
    import re
    from datetime import datetime

    scripts_dir = Path.home() / ".o3b" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime("%m%d_%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", name)
    path = scripts_dir / f"{ts}_{safe}.sh"
    path.write_text(content)
    return path


def _make_sbatch_script(cfg, job_name: str, env_vars: dict, remote_setup_script: str) -> str:
    """Return a complete sbatch script string built from platform config values."""

    node_count      = int(cfg.get("node_count", 1))
    gpu_count       = int(cfg.get("gpu_count_per_node", 1))
    cpu_count_gpu   = int(cfg.get("cpu_count_per_gpu", 8))
    # one task per node holds the node's whole GPU allocation (torchrun forks a
    # process per GPU inside it), so the task needs cpu_count_per_gpu cores for
    # each of them — dataloader workers scale with the GPU count
    cpu_count       = cpu_count_gpu * gpu_count
    ram_per_cpu     = cfg.get("ram_per_cpu", "5gb")
    walltime        = cfg.get("walltime", "24:00:00")
    partition       = cfg.get("partition", None)
    nodes_exclude   = cfg.get("nodes_exclude", None)
    restart         = cfg.get("restart_upon_fail", False)
    # JSC (juwels/jupiter) rejects every job without a budget account; the LMB
    # cluster has no accounting and leaves this unset.
    account         = cfg.get("account", None)
    # JSC allocates whole nodes and rejects --mem; ram_per_cpu is then advisory
    exclusive_nodes = str(cfg.get("exclusive_nodes", False)).lower() in ("true", "1", "yes")
    # path_home is defined in the custom overlay; fall back to path_ws
    path_home       = cfg.get("path_home", cfg.get("path_ws", "/tmp"))

    total_mem = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    optional = {
        "account":        f"#SBATCH --account={account}"          if account       else "",
        "requeue":        "#SBATCH --requeue"                    if restart       else "",
        "partition":      f"#SBATCH --partition {partition}"     if partition     else "",
        "nodes_exclude":  f"#SBATCH --exclude {nodes_exclude}"   if nodes_exclude else "",
    }

    env_block = "\n".join(f"export {k}={v!r}" for k, v in env_vars.items())

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH --nodes {node_count}",
        "#SBATCH --ntasks-per-node 1",
        f"#SBATCH --time {walltime}",
        f"#SBATCH --gres gpu:{gpu_count}",
        f"#SBATCH --cpus-per-task {cpu_count}",
        "#SBATCH --open-mode=append",
        f"#SBATCH -o {path_home}/slurm_jobs/%x_%j.o",
        "#SBATCH --mail-type=FAIL",
        "#SBATCH --signal=B:SIGUSR1@60",
    ]
    if not exclusive_nodes:
        lines.append(f"#SBATCH --mem {total_mem}")
    for v in optional.values():
        if v:
            lines.append(v)

    # Multi-node: the run script has to execute once per node (each invocation
    # starts the node's torchrun with node_rank=$SLURM_PROCID — see the
    # distributed block in `_srun_env_lines`).  A single node needs no srun:
    # sbatch already put us on it, and torchrun covers its GPUs from there.
    if node_count > 1:
        launch = (f"srun --nodes={node_count} --ntasks={node_count} "
                  f"--ntasks-per-node=1 bash {remote_setup_script}")
    else:
        launch = f"bash {remote_setup_script}"

    lines += [
        "",
        "set -euo pipefail",
        "",
        env_block,
        "",
        launch,
    ]
    return "\n".join(lines) + "\n"


def _find_setup_script(local_repo_root: str = "") -> Path:
    """Locate setup_slurm.sh, which ships with o3b (<o3b>/setup/setup_slurm.sh).

    It used to live in the superproject (housecorr3d/setup/), so that location is
    still accepted as a fallback for checkouts whose o3b submodule predates the
    move. The script itself is repo-agnostic — it installs whatever REPO_URL /
    REPO_NAME it is handed.
    """
    candidates = [Path(__file__).resolve().parents[2] / "setup" / "setup_slurm.sh"]
    if local_repo_root:
        candidates.append(Path(local_repo_root) / "setup" / "setup_slurm.sh")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Setup script not found. Looked in:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def _resolve_env_layout(cfg, repo_path: str, repo_name: str = "") -> dict:
    """Resolve the python environment a platform installs into and runs from.

    With ``use_conda: False`` (default) that is a venv inside the repo, with
    ``use_conda: True`` a conda env under ``<path_conda>/envs/``. The same
    values are exported to setup_slurm.sh (so it installs there) and used by
    `_srun_env_lines` (so `o3b run` / `o3b runi` activate the same env), which
    is why the naming lives in one place.

    Returns use_conda / path_conda / cuda_version / env_path plus ``env_vars``,
    the fragment to merge into the setup script's environment.
    """
    import os

    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps           = list(cfg.get("deps", []) or [])
    deps_tag       = "_".join(sorted(deps)) if deps else ""
    use_conda      = str(cfg.get("use_conda", False)).lower() in ("true", "1", "yes")
    path_ws        = cfg.get("path_ws", "") or ""
    path_conda     = str(cfg.get("path_conda", "") or (f"{path_ws}/miniconda3" if path_ws else ""))
    # conda ships its own cuda-toolkit; without an explicit cuda_version take the
    # one named by the system install (…/cuda-12.4 → 12.4)
    cuda_version   = str(cfg.get("cuda_version", "") or
                         os.path.basename(path_cuda).replace("cuda-", ""))

    # Always from cuda_version, which defaults to the basename of path_cuda and
    # so keeps the LMB tag at cu124. Deriving it from the basename directly
    # breaks on JSC, where the install is .../CUDA/12 (nvcc 12.6) rather than
    # .../cuda-12.4 -- that yielded "cu12" and a 404 pytorch index URL.
    cuda_tag  = "cu" + cuda_version.replace(".", "")
    py_tag    = "py" + python_version.replace(".", "")
    torch_tag = "torch" + ".".join(torch_version.split(".")[:2]).replace(".", "")
    env_tag   = f"{py_tag}_{cuda_tag}_{torch_tag}" + (f"_{deps_tag}" if deps_tag else "")

    if use_conda:
        name     = f"{repo_name or Path(repo_path).name}_{env_tag}"
        env_path = f"{path_conda}/envs/{name}" if path_conda else ""
    else:
        env_path = f"{repo_path}/venv_{env_tag}" if repo_path else ""

    return {
        "use_conda":    use_conda,
        "path_conda":   path_conda,
        "cuda_version": cuda_version,
        "cuda_tag":     cuda_tag,
        "env_path":     env_path,
        "env_vars": {
            "USE_CONDA":      "true" if use_conda else "false",
            "PATH_CONDA":     path_conda,
            "CUDA_VERSION":   cuda_version,
            "CONDA_ENV_PATH": env_path if use_conda else "",
            "VENV_PATH":      "" if use_conda else env_path,
        },
    }


def _credential(cfg, path: str, default: str = "") -> str:
    """Resolve ``credentials.<path>``, treating the unset placeholder as absent.

    credentials/default.yaml ships every key as the literal ``"..."`` so the
    file documents what can be set. Handing that string on to a tool as if it
    were a secret is worse than passing nothing: it shadows the credential the
    tool would otherwise have found (a ~/.netrc entry, an env var) and fails
    with a confusing auth error instead of a missing-credential one.
    """
    from omegaconf import OmegaConf

    value = str(OmegaConf.select(cfg, f"credentials.{path}", default="") or "")
    return default if value in ("", "...", "None") else value


def _run_platform_setup(args):
    import re
    import subprocess
    from omegaconf import OmegaConf, open_dict

    platform = args.platform
    # `--pull-only` (and `o3b bench rrun --pull`, which calls in here): refresh
    # the remote checkout and stop, skipping the env/dependency install. The
    # whole point is that it is cheap enough to run before every submission.
    pull_only = bool(getattr(args, "pull_only", False))
    # `--creds-only`: same early exit, but without the pull — re-ship the
    # credentials onto the checkout that is already there.
    creds_only = bool(getattr(args, "creds_only", False))
    short_setup = pull_only or creds_only

    print(f"Loading platform config '{platform}'…")
    cfg, configs_dir = _load_platform_config(platform)

    with open_dict(cfg):
        cfg.setup = True

    # The dump carries credentials.*.token in cleartext. A full setup is a rare,
    # deliberate act where seeing the resolved config is worth it; the short
    # forms run routinely, so they print a one-line summary instead (below, once
    # the ssh host has been validated).
    if not short_setup:
        print(OmegaConf.to_yaml(cfg))

    ssh_host = cfg.get("ssh")
    # ssh: False (or unset) → run the setup script locally instead of over SSH/SLURM.
    local_setup = (not ssh_host) or ssh_host is False

    # The local path deliberately never pulls (it would clobber the working tree
    # you are editing), so a pull-only local setup could only ever be a no-op.
    if short_setup and local_setup:
        what = "pull" if pull_only else "install credentials on"
        print(f"ERROR: platform '{platform}' has no ssh host — there is no remote "
              f"checkout to {what}.", file=sys.stderr)
        raise SystemExit(2)
    if short_setup:
        label = "Pull-only" if pull_only else "Credentials-only"
        print(f"{label} setup on {ssh_host} "
              f"(branch {cfg.get('branch', 'main')}, {cfg.get('path_ws', '')})")

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    # setup_slurm.sh reads one INSTALL_<DEP> flag per dep (INSTALL_MORPHEUS, …)
    # and defaults each to false, so only the enabled ones need exporting.
    install_flags        = {f"INSTALL_{dep.upper()}": "true" for dep in deps}
    branch         = cfg.get("branch", "main")
    pull           = cfg.get("pull", True)
    pull_submodules  = cfg.get("pull_submodules", True)
    # asking for a pull-only setup *is* asking to pull, whatever the config says
    if pull_only:
        pull = pull_submodules = True
    # …and --creds-only is the opposite: leave the checkout exactly as it is
    if creds_only:
        pull = pull_submodules = False
    skip_submodules  = " ".join(str(s) for s in list(cfg.get("skip_submodules", []) or []))
    # Lmod modules to load before anything else (JSC systems have no usable
    # system python/git/CUDA); empty on clusters that need none.
    modules          = " ".join(str(m) for m in list(cfg.get("modules", []) or []))
    # CUDA archs for extension builds. Unset = torch auto-detects, which crashes
    # on Grace-Hopper (sm_90a is unparseable by torch 2.6's _get_cuda_arch_flags).
    torch_arch_list  = str(cfg.get("torch_cuda_arch_list", "") or "")
    # torch.hub cache: JSC homes are quota-tight and compute nodes cannot
    # download, so point it at shared storage and pre-populate it during setup.
    torch_home       = str(cfg.get("path_torch_home", "") or "")
    warm_torch_hub   = " ".join(str(m) for m in list(cfg.get("warm_torch_hub", []) or []))
    username       = cfg.get("username", "")
    path_home      = cfg.get("path_home", path_ws)
    # run the setup on the login node instead of submitting it as a batch job:
    # required wherever compute nodes have no internet access (JSC)
    setup_on_login = str(cfg.get("setup_on_login", False)).lower() in ("true", "1", "yes")

    # Walk up from __file__ to find the outermost git repo (the repo that
    # contains o3b as a submodule) via --show-superproject-working-tree.
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            text=True,
            cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        repo_name = Path(local_repo_root).name
    except subprocess.CalledProcessError:
        repo_name = "housecorr3d"
        local_repo_root = str(Path.cwd())

    # The remote URL stays token-free: authentication on the cluster goes through
    # the GITHUB_TOKEN-backed credential helper installed by the setup script.
    # Embedding the token here would freeze it inside the clone's .git/config,
    # where a later rotation could never reach it.
    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    # wandb reads ~/.netrc, not the credentials yaml, so the key is handed to
    # setup_slurm.sh to write that entry on the remote
    wandb_api_key = _credential(cfg, "wandb.api_key")
    wandb_host    = _credential(cfg, "wandb.host", "api.wandb.ai")
    try:
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True, cwd=local_repo_root,
        ).strip()
        # Convert SSH → HTTPS if needed: git@github.com:Org/Repo → https://github.com/Org/Repo
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        # Strip any existing token
        plain = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = Path(local_repo_root).name

    setup_script_local = _find_setup_script(local_repo_root)

    # One directory per submission. Every run used to scp over the same
    # $PATH_WS/setup_slurm.sh, which silently corrupts a setup job that is still
    # running: bash reads a script incrementally, so replacing the file under a
    # running job resumes it at a stale byte offset, mid-line -- the symptom is a
    # nonsense "line NNN: p: command not found" and exit 127, minutes into the
    # install. The timestamp is shared with the local ~/.o3b/scripts copies so
    # the two can be correlated.
    from datetime import datetime
    ts = datetime.now().strftime("%m%d_%H%M%S")

    remote_run_dir = f"{path_ws}/setup_runs/{ts}"
    remote_setup   = f"{remote_run_dir}/setup_slurm.sh"
    remote_sbatch  = f"{remote_run_dir}/setup_slurm_job.sh"

    # credentials/custom/*.yaml is gitignored, so the remote clone never carries
    # it and every credential resolves to the "..." placeholder in
    # credentials/default.yaml. Ship the local files alongside the setup script;
    # setup_slurm.sh copies them into the repo once the o3b submodule is checked
    # out (the destination lives inside it, so it cannot exist before that).
    cred_src_dir  = configs_dir / "credentials" / "custom"
    cred_files    = [p for p in sorted(cred_src_dir.glob("*.yaml"))
                     if not p.name.endswith("_template.yaml")]
    remote_cred_dir  = f"{remote_run_dir}/credentials"
    try:
        cred_dest_rel = str(cred_src_dir.relative_to(local_repo_root))
    except ValueError:
        # o3b resolved from outside the checkout (e.g. site-packages) — nothing
        # to place relative to the remote repo, so skip the copy.
        cred_files = []
        cred_dest_rel = ""

    def _scp(local, remote):
        target = f"{ssh_host}:{remote}"
        if username:
            target = f"{username}@{ssh_host}:{remote}"
        print(f"Copying {local} → {target}")
        subprocess.run(["scp", str(local), target], check=True)

    # venv (default) or conda env — same resolution as `o3b run` / `o3b runi`
    env_layout = _resolve_env_layout(cfg, f"{path_ws}/{repo_name}", repo_name)

    # Build sbatch wrapper with #SBATCH headers from the platform config.
    # Only LMB reaches the internet through tfproxy; JSC routes directly and a
    # proxy set there breaks every outbound request, so this is per platform.
    _proxy = cfg.get("http_proxy", "") or ""
    env_vars = {
        "PATH_WS":         path_ws,
        "PATH_CUDA":       path_cuda,
        "PYTHON_VERSION":  python_version,
        "TORCH_VERSION":   torch_version,
        **install_flags,
        **env_layout["env_vars"],
        "DEPS_TAG":              deps_tag,
        "REPO_URL":        repo_url,   # housecorr3d HTTPS URL, token-free
        "REPO_NAME":       repo_name,  # derived from remote URL, e.g. HouseCorr3Dv2
        "GITHUB_TOKEN":    token,
        "BRANCH":          branch,
        "PULL":            "true" if pull else "false",
        "PULL_SUBMODULES": "true" if pull_submodules else "false",
        "PULL_ONLY":       "true" if pull_only else "false",
        "CREDS_ONLY":      "true" if creds_only else "false",
        # setup_slurm.sh turns this into a ~/.netrc entry, which is what the
        # wandb CLI/library actually read — the credentials yaml it also installs
        # is only ever consulted by o3b itself. Empty leaves ~/.netrc alone.
        "WANDB_API_KEY":   wandb_api_key,
        "WANDB_HOST":      wandb_host,
        "SKIP_SUBMODULES": skip_submodules,
        "MODULES":         modules,
        **({"TORCH_CUDA_ARCH_LIST": torch_arch_list} if torch_arch_list else {}),
        **({"TORCH_HOME": torch_home} if torch_home else {}),
        **({"WARM_TORCH_HUB": warm_torch_hub} if warm_torch_hub else {}),
        "CREDENTIALS_SRC":  remote_cred_dir if cred_files else "",
        "CREDENTIALS_DEST": cred_dest_rel if cred_files else "",
        "HTTP_PROXY":      _proxy,
        "HTTPS_PROXY":     _proxy,
        "http_proxy":      _proxy,
        "https_proxy":     _proxy,
    }
    # ── local setup (ssh: False) ──────────────────────────────────────────────
    # Run the setup script in-place on the existing local repo + venv. No clone,
    # no pull/submodule-update (so the working tree is left untouched), no SLURM.
    if local_setup:
        import os
        local_root = Path(local_repo_root)
        local_env = {
            **env_vars,
            "PATH_WS":         str(local_root.parent),  # REPO_PATH = PATH_WS/REPO_NAME
            "REPO_NAME":       local_root.name,          #            = local_root
            "REPO_URL":        "",                       # repo already present → skip clone
            "PULL":            "false",                  # don't disturb the local working tree
            "PULL_SUBMODULES": "false",
            "CREDENTIALS_SRC": "",                       # already in place locally
            "CREDENTIALS_DEST": "",
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": "",
        }
        if env_layout["use_conda"]:
            # the conda env lives under path_conda, not in the repo — rename it
            # after the local checkout directory
            local_env.update(_resolve_env_layout(cfg, str(local_root), local_root.name)["env_vars"])
        else:
            venv_path = local_root / "venv"
            if venv_path.is_dir():
                local_env["VENV_PATH"] = str(venv_path)  # reuse the existing venv
        local_copy = _save_script_locally(f"setup_{repo_name}_script", setup_script_local.read_text())
        print(f"Local setup (ssh: {ssh_host!r}) — running {setup_script_local} in {local_root}")
        print(f"  saved locally: {local_copy}")
        subprocess.run(
            ["bash", str(setup_script_local)],
            env={**os.environ, **local_env},
            check=True,
        )
        return

    # ── login-node setup (setup_on_login: true) ───────────────────────────────
    # JSC compute nodes have no route to the internet, so the usual "submit the
    # setup as a batch job" model cannot clone or pip-install there at all. The
    # login nodes do have connectivity (and, on JUPITER, the same Grace-Hopper
    # arch as the compute nodes, so CUDA extensions build correctly). Run the
    # script over ssh instead of sbatch, streaming its output.
    if setup_on_login:
        runner = "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "\n".join(f"export {k}={v!r}" for k, v in env_vars.items()),
            "",
            f"bash {remote_setup}",
        ]) + "\n"
        local_runner = _save_script_locally(f"setup_{repo_name}_login", runner, ts)
        print(f"  saved locally: {local_runner}")

        subprocess.run(
            ["ssh", ssh_host, f"mkdir -p {remote_run_dir} && chmod 700 {remote_run_dir}"],
            check=True,
        )
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
            tmp.write(runner)
            tmp_runner = tmp.name
        remote_runner = f"{remote_run_dir}/setup_login.sh"
        try:
            _scp(setup_script_local, remote_setup)
            _scp(tmp_runner, remote_runner)
        finally:
            Path(tmp_runner).unlink(missing_ok=True)

        if cred_files:
            subprocess.run(
                ["ssh", ssh_host, f"mkdir -p {remote_cred_dir} && chmod 700 {remote_cred_dir}"],
                check=True,
            )
            for cred in cred_files:
                _scp(cred, f"{remote_cred_dir}/{cred.name}")
            subprocess.run(["ssh", ssh_host, f"chmod 600 {remote_cred_dir}/*.yaml"], check=True)

        # keep the canonical copy fresh for `o3b platform setupi`; mv, not scp,
        # so a concurrent run reading it is not truncated mid-line
        print(f"Running setup on {ssh_host} (login node)…")
        remote_cmd = (
            f"chmod +x {remote_setup} {remote_runner} && "
            f"cp {remote_setup} {path_ws}/.setup_slurm.sh.new && "
            f"mv -f {path_ws}/.setup_slurm.sh.new {path_ws}/setup_slurm.sh && "
            f"bash {remote_runner}"
        )
        subprocess.run(["ssh", ssh_host, remote_cmd], check=True)
        return

    sbatch_script = _make_sbatch_script(
        cfg,
        job_name=f"setup_{repo_name}",
        env_vars=env_vars,
        remote_setup_script=remote_setup,
    )

    # Save both scripts locally under the same timestamp as the remote run dir
    local_setup_copy = _save_script_locally(f"setup_{repo_name}_script", setup_script_local.read_text(), ts)
    local_sbatch     = _save_script_locally(f"setup_{repo_name}_sbatch", sbatch_script, ts)
    print(f"  saved locally: {local_setup_copy}")
    print(f"  saved locally: {local_sbatch}")

    # 0700 on the run dir: the staged credentials below are plaintext tokens and
    # PATH_WS is group-readable on the cluster.
    subprocess.run(
        ["ssh", ssh_host, f"mkdir -p {remote_run_dir} && chmod 700 {remote_run_dir}"],
        check=True,
    )

    # Write sbatch script to a temp file and SCP both scripts
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
        tmp.write(sbatch_script)
        tmp_path = tmp.name

    try:
        _scp(setup_script_local, remote_setup)
        _scp(tmp_path, remote_sbatch)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if cred_files:
        subprocess.run(
            ["ssh", ssh_host, f"mkdir -p {remote_cred_dir} && chmod 700 {remote_cred_dir}"],
            check=True,
        )
        for cred in cred_files:
            _scp(cred, f"{remote_cred_dir}/{cred.name}")
        subprocess.run(["ssh", ssh_host, f"chmod 600 {remote_cred_dir}/*.yaml"], check=True)

    # Ensure output log directory exists, then submit via sbatch.
    # The canonical $PATH_WS/setup_slurm.sh is still read by the `setup: true`
    # job preamble and by `o3b platform setupi`, so keep refreshing it -- but via
    # mv, which swaps the directory entry. A job already executing the old file
    # keeps reading that inode intact instead of resuming mid-line in the new one.
    remote_cmd = (
        f"mkdir -p {path_home}/slurm_jobs && "
        f"chmod +x {remote_setup} {remote_sbatch} && "
        f"cp {remote_setup} {path_ws}/.setup_slurm.sh.new && "
        f"mv -f {path_ws}/.setup_slurm.sh.new {path_ws}/setup_slurm.sh && "
        f"sbatch {remote_sbatch}"
    )
    print(f"Submitting setup job on {ssh_host}…")
    subprocess.run(["ssh", ssh_host, remote_cmd], check=True)


def _maybe_pull_platform(args, platform: str) -> None:
    """Honour ``--pull``: refresh the remote checkout before using it.

    Both `bench rrun` and `platform runi` ship the job/shell preamble from the
    *local* tree while every line of Python comes from the remote checkout. On a
    setup_on_login platform the preamble is forced to PULL=false (compute nodes
    have no route to github), so that checkout only ever moves during setup —
    which is how a job silently ends up running whatever was last pulled.

    Failures propagate: better to stop than to submit against a stale tree.
    """
    if not getattr(args, "pull", False):
        return
    from types import SimpleNamespace

    print(f"--pull: refreshing the {platform} checkout…")
    _run_platform_setup(SimpleNamespace(platform=platform, pull_only=True))
    print()


def _fetch_jobs(ssh_host: str, username: str, hours: float = 2.0) -> list:
    """Return job list from sacct for the last *hours* hours as a list of dicts."""
    import subprocess
    fields      = ["JobID", "JobName", "State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList"]
    start_expr  = f"$(date -d '{hours} hours ago' +'%Y-%m-%dT%H:%M:%S')"
    cmd = (
        f"sacct --starttime={start_expr}"
        f" --format={','.join(fields)!r}"
        f" --parsable2 --allocations"
    )
    if username:
        cmd += f" -u {username}"
    result = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True, check=True)
    lines  = [l for l in result.stdout.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    jobs = [dict(zip(fields, row.split("|"))) for row in lines[1:]]
    jobs.sort(key=lambda j: int(j.get("JobID", 0) or 0), reverse=True)
    return jobs


def _open_log(ssh_host: str, path_home: str, job: dict) -> None:
    """Open the job log in less via ssh -t. Prints an error if the file is missing."""
    import subprocess
    log_path = f"{path_home}/slurm_jobs/{job['JobName']}_{job['JobID']}.o"
    check = subprocess.run(
        ["ssh", ssh_host, f"test -f {log_path!r} && echo yes || echo no"],
        capture_output=True, text=True,
    )
    if check.stdout.strip() != "yes":
        print(f"\nLog not found: {log_path}")
        print("(Jobs not submitted via `o3b platform setup` may write logs elsewhere.)")
        input("Press Enter to return…")
        return
    subprocess.run(["ssh", "-t", ssh_host, f"less +G {log_path!r}"])


def _kill_job(ssh_host: str, job: dict) -> None:
    """Cancel a SLURM job via scancel after confirmation."""
    import subprocess
    job_id = job["JobID"]
    print(f"\nscancel {job_id} ({job['JobName']})  [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        return
    result = subprocess.run(["ssh", ssh_host, f"scancel {job_id}"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"scancel failed: {result.stderr.strip()}")
    else:
        print(f"Job {job_id} cancelled.")
    input("Press Enter to continue…")


def _overview_tui(stdscr, jobs: list, ssh_host: str, title: str):
    """
    Curses TUI for job selection.
    Returns: ('view', job_dict) | ('refresh',) | ('quit',)
    """
    import curses

    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK,  curses.COLOR_CYAN)   # selected row
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_WHITE)  # column header
    curses.init_pair(3, curses.COLOR_RED,    -1)                  # FAILED / ERROR
    curses.init_pair(4, curses.COLOR_GREEN,  -1)                  # COMPLETED
    curses.init_pair(5, curses.COLOR_YELLOW, -1)                  # RUNNING / other
    curses.init_pair(6, curses.COLOR_WHITE,  curses.COLOR_BLUE)   # title / status bar

    COLS = [
        ("JobID",     10, ">"),
        ("JobName",   60, "<"),
        ("State",     14, "<"),
        ("ExitCode",   8, ">"),
        ("Elapsed",   10, "<"),
        ("Start",     19, "<"),
        ("Partition", 20, "<"),
    ]

    def fmt_row(job):
        parts = []
        for field, w, align in COLS:
            val = str(job.get(field, ""))
            val = val[:w] if align == "<" else val
            parts.append(f"{val:{align}{w}}")
        return "  ".join(parts)

    def state_attr(state):
        if any(s in state for s in ("FAIL", "ERROR", "TIMEOUT", "OUT_OF")):
            return curses.color_pair(3)
        if "COMPLET" in state:
            return curses.color_pair(4)
        return curses.color_pair(5)

    current = 0
    offset  = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── title bar ──────────────────────────────────────────────
        bar = f" {title}  [R] refresh  [K] kill  [Q] quit "
        stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        stdscr.addstr(0, 0, bar[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

        # ── column header ──────────────────────────────────────────
        hdr_job  = {f: f for f, *_ in COLS}
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, fmt_row(hdr_job)[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # ── job rows ───────────────────────────────────────────────
        list_h  = h - 3          # title + header + status bar
        visible = jobs[offset: offset + list_h]
        for i, job in enumerate(visible):
            y   = i + 2
            idx = i + offset
            row = fmt_row(job)[:w - 1]
            if idx == current:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(y, 0, row.ljust(w - 1))
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                attr = state_attr(job.get("State", ""))
                stdscr.attron(attr)
                stdscr.addstr(y, 0, row)
                stdscr.attroff(attr)

        # ── status bar ─────────────────────────────────────────────
        count = f"[{current + 1}/{len(jobs)}]" if jobs else "[0/0]"
        status = f" {count}  ↑↓ navigate   Enter / L : show logs   K kill   R refresh   Q quit"
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(h - 1, 0, status[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6))

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):
            return ("quit",)
        elif key in (ord("r"), ord("R")):
            return ("refresh",)
        elif key == curses.KEY_UP:
            if current > 0:
                current -= 1
                if current < offset:
                    offset -= 1
        elif key == curses.KEY_DOWN:
            if current < len(jobs) - 1:
                current += 1
                if current >= offset + list_h:
                    offset += 1
        elif key in (ord("\n"), ord("l"), ord("L"), curses.KEY_ENTER) and jobs:
            return ("view", jobs[current])
        elif key in (ord("k"), ord("K")) and jobs:
            return ("kill", jobs[current])


def _run_platform_status(args):
    import curses
    import subprocess
    from omegaconf import OmegaConf

    platform = args.platform
    cfg, _   = _load_platform_config(platform)

    ssh_host  = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    username  = cfg.get("username", "")
    path_home = cfg.get("path_home", cfg.get("path_ws", ""))

    if args.configs:
        print("=" * 60)
        print(f"Platform config: {platform}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        input("Press Enter to open the job overview…")

    # ── active jobs: show as plain text before entering the TUI ─────
    squeue_fmt = "%.10i %.12P %.30j %.10u %.10T %.12M %.12l %.5D %R"
    squeue_cmd = f"squeue --format={squeue_fmt!r}"
    if username:
        squeue_cmd += f" -u {username}"
    print("=" * 60)
    print(f"Active jobs on {ssh_host}" + (f" (user: {username})" if username else ""))
    print("=" * 60)
    subprocess.run(["ssh", ssh_host, squeue_cmd], check=True)
    print()

    # ── TUI loop ────────────────────────────────────────────────────
    hours = getattr(args, "hours", 2.0)
    hours_label = f"{hours:g} h"
    tui_title = f"SLURM overview · {ssh_host}" + (f" · {username}" if username else "") + f" · last {hours_label}"
    jobs = _fetch_jobs(ssh_host, username, hours=hours)

    while True:
        action = curses.wrapper(lambda scr: _overview_tui(scr, jobs, ssh_host, tui_title))

        if action[0] == "quit":
            break
        elif action[0] == "refresh":
            jobs = _fetch_jobs(ssh_host, username)
        elif action[0] == "view":
            _open_log(ssh_host, path_home, action[1])
        elif action[0] == "kill":
            _kill_job(ssh_host, action[1])
            jobs = _fetch_jobs(ssh_host, username)


def _mp_env_from_cfg(cfg) -> dict:
    """Environment the job preamble exports before it runs anything.

    Platform keys (see configs/platform/slurm.yaml):
      mp_sharing_strategy → MP_SHARING_STRATEGY  applied at `import o3b`
      mp_start_method     → MP_START_METHOD      applied at `import o3b`
      nofile_limit        → NOFILE_LIMIT         consumed by `_srun_env_lines`
                                                 as a `ulimit -n`, not exported
      env: {NAME: value}  → exported verbatim
    Absent keys are omitted, leaving the node's / torch's defaults in place.

    The free-form ``env:`` block is for variables a run has to set *before* the
    process starts, which is the only point at which some of them are still
    read — OMP_NUM_THREADS is fixed at the first torch import, and torchrun
    only defaults it to 1 when it is not already in the environment. Because
    `o3b bench rrun` merges an ablation's ``platform:`` block over the platform
    config, an ablation can set one per variant (morpheus/omp_num_threads)
    without a platform config per value.
    """
    keys = {"mp_sharing_strategy": "MP_SHARING_STRATEGY",
            "mp_start_method":     "MP_START_METHOD",
            "nofile_limit":        "NOFILE_LIMIT"}
    env = {var: str(cfg.get(key, None)) for key, var in keys.items()
           if cfg.get(key, None)}
    from omegaconf import OmegaConf
    extra = cfg.get("env", None)
    if extra:
        if OmegaConf.is_config(extra):
            extra = OmegaConf.to_container(extra, resolve=True)
        env.update({str(k): str(v) for k, v in dict(extra).items()})
    return env


def _platform_srun_context(platform: str):
    """Return the srun context for a platform.

    (ssh_host, srun_base, repo_path, env_path, path_cuda, path_ws,
     hf_datasets_cache, use_conda, path_conda, mp_env) — env_path is the venv
    directory or, with use_conda, the conda env prefix; mp_env is the
    multiprocessing / fd-limit environment from `_mp_env_from_cfg`.
    """
    import os, re, subprocess
    from omegaconf import OmegaConf

    cfg, _ = _load_platform_config(platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    partition     = cfg.get("partition", None)
    node_count    = int(cfg.get("node_count", 1))
    gpu_count     = int(cfg.get("gpu_count_per_node", 1))
    # cores (and hence memory) for the task's whole GPU allocation — see
    # `_make_sbatch_script`, which sizes the batch jobs the same way
    cpu_count     = int(cfg.get("cpu_count_per_gpu", 8)) * gpu_count
    ram_per_cpu   = cfg.get("ram_per_cpu", "5gb")
    walltime      = cfg.get("walltime", "24:00:00")
    nodes_exclude = cfg.get("nodes_exclude", None)
    account       = cfg.get("account", None)          # mandatory on JSC, unset at LMB
    proxy         = cfg.get("http_proxy", "") or ""   # empty = direct connection
    exclusive_nodes = str(cfg.get("exclusive_nodes", False)).lower() in ("true", "1", "yes")
    modules       = " ".join(str(m) for m in list(cfg.get("modules", []) or []))
    torch_arch_list = str(cfg.get("torch_cuda_arch_list", "") or "")
    # "offline" where the compute nodes cannot reach the wandb servers (JSC)
    wandb_mode    = str(cfg.get("wandb_mode", "") or "")
    total_mem     = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    install_flags        = {f"INSTALL_{dep.upper()}": "true" for dep in deps}
    setup          = "true" if cfg.get("setup", False) else "false"
    branch         = str(cfg.get("branch", "main"))
    pull           = str(cfg.get("pull", True)).lower()
    pull_subs      = str(cfg.get("pull_submodules", True)).lower()
    # The compute-node preamble cannot fetch where setup_on_login is set: that
    # flag means the compute nodes have no route to github (which is precisely
    # why setup runs on the login node), and a fetch there fails the whole job.
    # `o3b platform setup` keeps the checkout current instead. Kept separate
    # from the `pull` key itself, which setup on the login node still honours.
    if str(cfg.get("setup_on_login", False)).lower() in ("true", "1", "yes"):
        pull, pull_subs = "false", "false"
    skip_subs      = " ".join(str(s) for s in list(cfg.get("skip_submodules", []) or []))
    hf_datasets_cache = cfg.get("path_hf_datasets_cache", "") or ""

    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=local_repo_root,
        ).strip()
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        plain    = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain   # token-free; auth comes from the GITHUB_TOKEN credential helper
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = ""

    repo_path  = f"{path_ws}/{repo_name}" if (path_ws and repo_name) else path_ws
    env_layout = _resolve_env_layout(cfg, repo_path, repo_name)
    env_path   = env_layout["env_path"]

    srun = (
        f"srun"
        f" --nodes {node_count}"
        f" --ntasks-per-node 1"
        f" --gres gpu:{gpu_count}"
        f" --cpus-per-task {cpu_count}"
        f" --time {walltime}"
    )
    # JSC allocates (and charges) whole nodes and rejects --mem outright
    if not exclusive_nodes:
        srun += f" --mem {total_mem}"
    if partition:
        srun += f" --partition {partition}"
    if account:
        srun += f" --account={account}"
    if nodes_exclude:
        srun += f" --exclude {nodes_exclude}"
    if path_ws:
        srun += f" --chdir {path_ws}"

    srun += " --export=ALL"
    # The LMB cluster only reaches the internet through tfproxy; JSC nodes route
    # directly and setting a proxy there breaks every outbound request. Configured
    # per platform (`http_proxy:`), empty = no proxy.
    if proxy:
        srun += (
            f",HTTP_PROXY={proxy}"
            f",HTTPS_PROXY={proxy}"
            f",http_proxy={proxy}"
            f",https_proxy={proxy}"
        )
    srun += (
        f",PATH_WS={path_ws}"
        f",PATH_CUDA={path_cuda}"
        f",PYTHON_VERSION={python_version}"
        f",TORCH_VERSION={torch_version}"
        + "".join(f",{k}={v}" for k, v in install_flags.items()) +
        f",REPO_URL={repo_url}"
        f",REPO_NAME={repo_name}"
        f",SETUP={setup}"
        f",BRANCH={branch}"
        f",PULL={pull}"
        f",PULL_SUBMODULES={pull_subs}"
        f",SKIP_SUBMODULES={skip_subs}"
        f",GITHUB_TOKEN={token}"
        + "".join(f",{k}={v}" for k, v in env_layout["env_vars"].items()) +
        f",DEPS_TAG={deps_tag}"
    )
    if torch_arch_list:
        srun += f",TORCH_CUDA_ARCH_LIST={torch_arch_list}"
    if wandb_mode:
        srun += f",WANDB_MODE={wandb_mode}"
    # jobs must read the cache warmed by setup, not $HOME's default
    _torch_home = str(cfg.get("path_torch_home", "") or "")
    if _torch_home:
        srun += f",TORCH_HOME={_torch_home}"
    # Under conda the toolchain comes from the env (CONDA_PREFIX), exported by
    # the preamble once the env is active.
    if not env_layout["use_conda"]:
        srun += f",CUDA_HOME={path_cuda},CUDACXX={path_cuda}/bin/nvcc"

    return (ssh_host, srun, repo_path, env_path, path_cuda, path_ws, hf_datasets_cache,
            env_layout["use_conda"], env_layout["path_conda"], _mp_env_from_cfg(cfg),
            modules)


def _srun_env_lines(path_cuda: str, env_path: str, repo_path: str, path_ws: str,
                    hf_datasets_cache: str = "", use_conda: bool = False,
                    path_conda: str = "", mp_env: dict | None = None,
                    modules: str = "", node_count: int = 1,
                    gpu_count: int = 1, nccl_env: dict | None = None) -> list[str]:
    """Shell lines that run on the compute node before the actual command.

    Order: fd limit + multiprocessing env → CUDA env → conditional setup script
           → conditional pull/checkout → env activation (venv or conda)
           → cd into repo → distributed (torchrun) setup.
    The SETUP / PULL / PULL_SUBMODULES / BRANCH values come from the srun
    --export env vars so the same script works regardless of platform config.
    ``mp_env`` comes from `_mp_env_from_cfg`; it is emitted first so it applies
    to everything the job runs, interactive shell or batch command alike.

    ``node_count`` / ``gpu_count`` (per node) come from the platform config and
    its ablation overrides.  With more than one GPU in total the preamble ends
    with the whole distributed setup — rendezvous address and port, NCCL env,
    and the ``o3b_launch`` wrapper that runs a command under torchrun.  It
    lives here, in the shared preamble, so a batch job and an interactive
    ``o3b platform runi`` shell set up multi-GPU identically.
    """
    def _echo(msg: str, indent: str = "") -> str:
        return f'{indent}echo "[o3b-init $(date +%T)] {msg}"'

    lines = [
        _echo("sourcing ~/.bashrc"),
        # Site bashrc files are not `set -u` clean and the job script runs under
        # `set -euo pipefail`. JSC's /etc/bashrc opens with
        # `if [ -z "$BASHRCSOURCED" ]` on an unset variable, which aborts every
        # job on line 2 with "BASHRCSOURCED: unbound variable". Drop -u across
        # the sourcing and restore it only if it was on -- `o3b platform runi`
        # reuses this preamble for an interactive shell that never set it.
        "case $- in *u*) _o3b_had_u=1 ;; *) _o3b_had_u=0 ;; esac",
        "set +u",
        "[ -f ~/.bashrc ] && . ~/.bashrc",
        'if [ "${_o3b_had_u}" = "1" ]; then set -u; fi',
    ]
    mp_env = dict(mp_env or {})
    nofile = mp_env.pop("NOFILE_LIMIT", "")
    if nofile:
        # slurm propagates the submit host's soft limit (1024) to the job, well
        # below what DataLoader fd passing needs; the hard limit is 131072
        # and raising within it needs no privileges.  Non-fatal: some shells
        # refuse, and the run is still worth attempting.
        target = '"$(ulimit -Hn)"' if str(nofile) == "hard" else str(nofile)
        lines += [
            f"ulimit -n {target} 2>/dev/null || true",
            _echo("open file limit: $(ulimit -Sn) (hard $(ulimit -Hn))"),
        ]
    for key, value in mp_env.items():
        lines.append(f"export {key}={value}")
    if mp_env:
        # echo the exported values, not the ones we templated in, so the log
        # shows what the job actually runs with
        exported = " ".join(f"{key}=${{{key}}}" for key in mp_env)
        lines.append(_echo(f"environment: {exported}"))
    # With conda the CUDA toolchain lives inside the env and is exported after
    # activation below; pointing at the system install here would shadow it.
    if modules:
        # JSC systems provide python/git/CUDA only through Lmod. Not `set -e`
        # clean, and non-interactive shells may lack the init sourced by .bashrc.
        lines += [
            _echo(f"module load {modules}"),
            "command -v module >/dev/null 2>&1 || "
            "{ [ -f /usr/share/lmod/lmod/init/bash ] && . /usr/share/lmod/lmod/init/bash; }",
        ]
        lines += [f'module load {m} || echo "WARNING: module load {m} failed"'
                  for m in modules.split()]
    if not use_conda:
        # targets/<triple> follows the CPU: sbsa-linux on JUPITER's Grace nodes,
        # x86_64-linux elsewhere. Resolved on the compute node, not here.
        lines += [
            f"export CUDA_HOME={path_cuda}",
            f"export CUDACXX={path_cuda}/bin/nvcc",
            f"export PATH={path_cuda}/bin:$PATH",
            f"export LD_LIBRARY_PATH={path_cuda}/lib64:${{LD_LIBRARY_PATH:-}}",
            'case "$(uname -m)" in aarch64|arm64) _cuda_triple=sbsa-linux ;; '
            "*) _cuda_triple=x86_64-linux ;; esac",
            f"export CPATH=${{CPATH:-}}:{path_cuda}/targets/${{_cuda_triple}}/include",
            f"export LIBRARY_PATH=${{LIBRARY_PATH:-}}:{path_cuda}/targets/${{_cuda_triple}}/lib",
        ]
    if hf_datasets_cache:
        lines.append(f"export HF_DATASETS_CACHE={hf_datasets_cache}")
    # acquire the same directory lock used by setup_slurm.sh so concurrent
    # srun jobs don't race on git pull / submodule update / venv install
    if path_ws:
        lines += [
            _echo("waiting for setup_slurm.lock (held by another concurrent job?)"),
            f'exec 200>{path_ws}/setup_slurm.lock',
            f'flock -x 200',
            _echo("lock acquired"),
        ]
    # run full setup script (e.g. install deps) when SETUP=true
    if path_ws:
        lines += [
            f'if [ "${{SETUP:-false}}" = "true" ]; then',
            _echo("running setup_slurm.sh (SETUP=true)", "    "),
            f'    bash {path_ws}/setup_slurm.sh',
            _echo("setup_slurm.sh done", "    "),
            f'fi',
        ]
    # github auth: the helper stored in ~/.gitconfig references ${GITHUB_TOKEN}
    # instead of embedding it, so a rotated token takes effect on the next job.
    # Legacy `url."https://<token>@github.com/".insteadOf` sections carried the
    # token in the config *key*: rotating appended a second section, and git
    # breaks insteadOf ties by config order, so the oldest token kept winning.
    if repo_path:
        lines += [
            _echo("configuring github credentials"),
            # ~/.gitconfig and <repo>/.git/config are on shared storage, so every
            # task of a multi-node job (and any concurrent job) writes the same
            # file. git guards a write with a .lock sibling and fails outright if
            # it already exists -- under the job's `set -e` that killed whole
            # tasks, and torchrun then hung forever on the missing rank. The
            # writes are idempotent (all tasks write identical content), so a lost
            # race only means someone else already did it: retry, then give up
            # without failing the job. The setup_slurm.lock flock around this
            # block does not serialise across nodes on every filesystem.
            "_o3b_git() {",
            "    _i=0",
            '    while [ "${_i}" -lt 10 ]; do',
            '        if git "$@" 2>/dev/null; then return 0; fi',
            "        _i=$((_i + 1))",
            "        sleep 1",
            "    done",
            '    echo "WARNING: git $* failed after 10 attempts '
            '(concurrent job holding the .lock?)" >&2',
            "    return 0",
            "}",
            # must be exported: the credential helper runs as a git subprocess
            'export GITHUB_TOKEN="${GITHUB_TOKEN:-}"',
            "for _k in $(git config --global --name-only --get-regexp "
            "'^url\\..*github\\.com.*\\.insteadof$' 2>/dev/null || true); do",
            '    git config --global --remove-section "${_k%.insteadof}" 2>/dev/null || true',
            "done",
            'if [ -n "${GITHUB_TOKEN:-}" ]; then',
            '    _o3b_git config --global --replace-all credential."https://github.com".helper '
            "'"'!f() { [ "$1" = get ] || exit 0; echo username=x-access-token; '
            'echo "password=${GITHUB_TOKEN}"; }; f'"'",
            "fi",
            "export GIT_TERMINAL_PROMPT=0",
            # clones made by older revisions have the token baked into .git/config,
            # which makes git skip the credential helper entirely
            f'_url="$(git -C {repo_path} remote get-url origin 2>/dev/null || true)"',
            'case "${_url}" in',
            '    https://*@github.com/*)',
            f'        _o3b_git -C {repo_path} remote set-url origin '
            '"https://github.com/${_url#*@github.com/}" ;;',
            "esac",
        ]
    # checkout branch and pull when PULL=true
    if repo_path:
        lines += [
            f'if [ "${{PULL:-false}}" = "true" ]; then',
            _echo(f"git fetch (repo={repo_path}, proxy=${{HTTPS_PROXY:-none}})", "    "),
            f'    git -C {repo_path} fetch',
            _echo("git checkout ${BRANCH:-main}", "    "),
            f'    git -C {repo_path} checkout "${{BRANCH:-main}}"',
            _echo("git pull", "    "),
            f'    git -C {repo_path} pull',
            _echo("git pull done", "    "),
            f'fi',
            f'if [ "${{PULL_SUBMODULES:-false}}" = "true" ]; then',
            f'    git -C {repo_path} submodule sync --recursive',
            f'    if [ -z "${{SKIP_SUBMODULES:-}}" ]; then',
            _echo("submodule update --init --recursive (all submodules)", "        "),
            f'        git -C {repo_path} submodule update --init --recursive',
            f'    else',
            f'        git -C {repo_path} submodule init',
            f"        for sub in $(git -C {repo_path} submodule status | awk '{{print $2}}'); do",
            f'            _skip=false',
            f'            for s in ${{SKIP_SUBMODULES}}; do',
            f'                [ "$sub" = "$s" ] && _skip=true && break',
            f'            done',
            f'            if [ "$_skip" = "false" ]; then',
            _echo("submodule update --init --recursive -- $sub", "                "),
            f'                git -C {repo_path} submodule update --init --recursive -- "$sub"',
            f'            fi',
            f'        done',
            f'    fi',
            _echo("submodule update done", "    "),
            f'fi',
        ]
    if path_ws:
        lines += [
            f'exec 200>&-',
            _echo("lock released"),
        ]
    if env_path and use_conda:
        lines += [
            _echo(f"activating conda env {env_path}"),
            # conda's shell hook and activate scripts are not `set -u` clean
            "case $- in *u*) _o3b_had_u=1 ;; *) _o3b_had_u=0 ;; esac",
            "set +u",
            f'eval "$({path_conda}/bin/conda shell.bash hook)"',
            f"conda activate {env_path}",
            'if [ "${_o3b_had_u}" = "1" ]; then set -u; fi',
            "hash -r",   # a ~/.local/bin/pip|python may shadow the env's in the hash cache
            # the cuda-toolkit is installed inside the env (see setup_slurm.sh)
            "export CUDA_HOME=${CONDA_PREFIX}",
            "export CUDACXX=${CUDA_HOME}/bin/nvcc",
            "export PATH=${CUDA_HOME}/bin:$PATH",
            # conda uses lib/ + include/ where a system CUDA install uses lib64/
            "export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}",
            "export LIBRARY_PATH=${CONDA_PREFIX}/lib:${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}",
            "export CPATH=${CONDA_PREFIX}/include:${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}",
        ]
    elif env_path:
        lines += [
            _echo(f"activating venv {env_path}"),
            f"[ -d {env_path} ] && source {env_path}/bin/activate",
        ]
    if repo_path:
        lines.append(f"cd {repo_path}")

    # ── distributed (torchrun) setup ──────────────────────────────────────────
    # O3B_NNODES / O3B_NPROC_PER_NODE are what `o3b_launch` reads.  The whole
    # block is emitted whatever the counts are, so an interactive session can
    # go multi-GPU by exporting O3B_NPROC_PER_NODE before calling o3b_launch.
    # On one GPU it changes nothing: o3b_launch execs the command directly, and
    # MASTER_ADDR / MASTER_PORT alone create no process group — that needs the
    # WORLD_SIZE only torchrun exports (see o3b.ddp.init_distributed).
    lines += [
        f"export O3B_NNODES={int(node_count)}",
        f"export O3B_NPROC_PER_NODE={int(gpu_count)}",
        # torchrun sets OMP_NUM_THREADS=1 whenever it is unset, so every rank of
        # a multi-GPU job would run its CPU-side work (collate, mask distance
        # transforms, mesh ops) single-threaded while the 1-GPU baseline — which
        # never goes through torchrun — uses the whole allocation.  That alone
        # makes the two incomparable.  Split the cores slurm actually granted
        # over the ranks sharing them rather than trusting the requested count,
        # and yield to a value the caller already set.
        'if [ -z "${OMP_NUM_THREADS:-}" ]; then',
        '    _o3b_cpus="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"',
        '    OMP_NUM_THREADS=$(( _o3b_cpus / ${O3B_NPROC_PER_NODE:-1} ))',
        '    [ "${OMP_NUM_THREADS}" -lt 1 ] && OMP_NUM_THREADS=1',
        "    export OMP_NUM_THREADS",
        "fi",
    ]
    for key, value in (nccl_env or {}).items():
        lines.append(f"export {key}={value}")
    lines += [
        # every node runs this same preamble and they must agree on the
        # rendezvous without talking to each other first: the address is the
        # allocation's first node, the port is derived from the job id (a
        # random draw would differ per node). Both yield to a value already in
        # the environment.
        'if [ -z "${MASTER_ADDR:-}" ]; then',
        '    if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then',
        '        MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"',
        "    else",
        "        MASTER_ADDR=127.0.0.1",
        "    fi",
        "fi",
        'if [ -z "${MASTER_PORT:-}" ]; then',
        '    MASTER_PORT="$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))"',
        "fi",
        "export MASTER_ADDR MASTER_PORT",
        'if [ "$(( ${O3B_NNODES:-1} * ${O3B_NPROC_PER_NODE:-1} ))" -gt 1 ]; then',
        _echo("distributed: ${O3B_NNODES} node(s) x ${O3B_NPROC_PER_NODE} gpu(s), "
              "rendezvous ${MASTER_ADDR}:${MASTER_PORT}, "
              "node_rank ${SLURM_PROCID:-0}", "    "),
        "fi",
        # `o3b_launch <console-script> <args…>` runs the command under torchrun
        # when the job holds more than one GPU, and plainly otherwise. torchrun
        # wants a script path, hence the `command -v` — the console script in
        # the active env is a python file it can exec.
        "o3b_launch() {",
        '    if [ "$(( ${O3B_NNODES:-1} * ${O3B_NPROC_PER_NODE:-1} ))" -le 1 ]; then',
        '        "$@"',
        "        return",
        "    fi",
        '    local _bin; _bin="$(command -v "$1")" || { echo "o3b_launch: $1 not found" >&2; return 127; }',
        "    shift",
        '    torchrun --nnodes="${O3B_NNODES}" --nproc_per_node="${O3B_NPROC_PER_NODE}"'
        ' --node_rank="${SLURM_PROCID:-0}"'
        ' --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}"'
        ' "$_bin" "$@"',
        "}",
    ]

    lines.append(_echo("init done, dropping into shell"))
    return lines


def _run_platform_runi(args):
    import subprocess
    import uuid

    # before the allocation, so a failed pull costs no queue time and the shell
    # always lands in a checkout matching what was pushed
    _maybe_pull_platform(args, args.platform)

    (ssh_host, srun, repo_path, env_path, path_cuda, path_ws, hf_datasets_cache,
     use_conda, path_conda, mp_env, modules) = _platform_srun_context(args.platform)

    # Write a small activation script so bash --init-file can source it without
    # wrapping srun in a bash -c subshell (which breaks the PTY).
    init_lines = _srun_env_lines(path_cuda, env_path, repo_path, path_ws, hf_datasets_cache,
                                 use_conda=use_conda, path_conda=path_conda, mp_env=mp_env,
                                 modules=modules)
    remote_init = f"{path_ws}/.od3d_init" if path_ws else "~/.od3d_init"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_init}"],
        input="\n".join(init_lines), text=True, check=True,
    )

    tunnel_proc = None
    forward = getattr(args, "forward", None)
    if forward:
        local_port, _, remote_port = forward.partition(":")
        if not remote_port:
            raise ValueError(f"--forward must be LOCAL:REMOTE, got {forward!r}")
        job_name = f"o3b_runi_{uuid.uuid4().hex[:8]}"
        srun += f" --job-name {job_name}"
        tunnel_proc = _forward_port_once_running(ssh_host, job_name, local_port, remote_port)

    srun += f" --pty bash --init-file {remote_init}"
    print(f"Opening interactive session on {ssh_host} in {repo_path or path_ws or '~'}…")
    try:
        subprocess.run(["ssh", "-t", ssh_host, srun])
    finally:
        if tunnel_proc is not None:
            tunnel_proc.stop()


class _PortTunnel:
    """Handle to a background port-forward thread + the ssh -L process it starts."""

    def __init__(self):
        self.proc: "subprocess.Popen | None" = None
        self._stop = False

    def stop(self):
        self._stop = True
        if self.proc is not None:
            self.proc.terminate()


def _forward_port_once_running(ssh_host: str, job_name: str, local_port: str, remote_port: str) -> "_PortTunnel":
    """Poll squeue for *job_name*'s node, then open `ssh -L local:node:remote` in the background.

    Runs in a daemon thread so the caller's interactive srun session isn't blocked
    waiting for the job to start.
    """
    import shlex
    import subprocess
    import threading
    import time

    handle = _PortTunnel()
    # ssh joins trailing argv elements with plain spaces before handing them to the
    # remote shell, so a value containing a space (the -o format string) must be
    # quoted ourselves and passed as a single already-assembled command string.
    remote_cmd = (
        f"squeue -h -n {shlex.quote(job_name)} -o {shlex.quote('%N %T')}"
    )

    def _watch():
        node = None
        for _ in range(120):  # ~2 minutes
            if handle._stop:
                return
            time.sleep(1)
            try:
                out = subprocess.check_output(
                    ["ssh", ssh_host, remote_cmd],
                    text=True, timeout=10,
                ).strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
            if not out:
                continue
            parts = out.split()
            if len(parts) == 2 and parts[1] == "RUNNING" and parts[0] not in ("", "(None)"):
                node = parts[0]
                break
        if handle._stop:
            return
        if node is None:
            print(f"WARNING: timed out waiting for job {job_name} to start — no port forwarded.")
            return
        print(f"Forwarding localhost:{local_port} -> {node}:{remote_port} (via {ssh_host})")
        handle.proc = subprocess.Popen(
            ["ssh", "-N", "-L", f"{local_port}:{node}:{remote_port}", ssh_host],
        )

    threading.Thread(target=_watch, daemon=True).start()
    return handle


def _run_platform_setupi(args):
    """Interactive setup: open a compute-node shell, copy setup_slurm.sh, and print it."""
    import re
    import subprocess

    (ssh_host, srun, repo_path, env_path, path_cuda, path_ws, hf_datasets_cache,
     use_conda, path_conda, mp_env, modules) = _platform_srun_context(args.platform)

    cfg, _ = _load_platform_config(args.platform)
    username = cfg.get("username", "")

    # the superproject is only the fallback location (same logic as _run_platform_setup)
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
    except subprocess.CalledProcessError:
        local_repo_root = str(Path.cwd())

    setup_script_local = _find_setup_script(local_repo_root)

    remote_setup = f"{path_ws}/setup_slurm.sh" if path_ws else "~/setup_slurm.sh"

    def _scp(local, remote):
        target = f"{ssh_host}:{remote}"
        if username:
            target = f"{username}@{ssh_host}:{remote}"
        print(f"Copying {local} → {target}")
        subprocess.run(["scp", str(local), target], check=True)

    # scp truncates in place, which would corrupt a setup job currently reading
    # this path; stage next to it and swap the directory entry instead.
    _scp(setup_script_local, f"{remote_setup}.new")
    subprocess.run(["ssh", ssh_host, f"mv -f {remote_setup}.new {remote_setup}"], check=True)

    init_lines = _srun_env_lines(path_cuda, env_path, repo_path, path_ws, hf_datasets_cache,
                                 use_conda=use_conda, path_conda=path_conda, mp_env=mp_env,
                                 modules=modules)
    init_lines += [
        "",
        f'echo "================================================================"',
        f'echo "  Setup script : {remote_setup}"',
        f'echo "  To execute   : bash {remote_setup}"',
        f'echo "================================================================"',
        f'echo ""',
        f'cat {remote_setup}',
        f'echo ""',
        f'echo "================================================================"',
        f'echo "  Run: bash {remote_setup}"',
        f'echo "================================================================"',
    ]

    remote_init = f"{path_ws}/.od3d_setupi_init" if path_ws else "~/.od3d_setupi_init"
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_init}"],
        input="\n".join(init_lines), text=True, check=True,
    )

    srun += f" --pty bash --init-file {remote_init}"
    print(f"Opening interactive setup session on {ssh_host}…")
    subprocess.run(["ssh", "-t", ssh_host, srun])


def _run_platform_run_cmd(platform: str, command: str, job_name: str | None = None) -> None:
    """Submit *command* as a queued sbatch job on the platform and return once submitted.

    Uses sbatch (not srun) so the job is queued independently of this process —
    closing the terminal/SSH session does not kill it. Output goes to the job's
    log file under <path_home>/slurm_jobs/, not to this terminal.
    """
    import re

    if job_name is None:
        job_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", command).strip("_")[:60] or "o3b_run"
    _run_bench_sbatch_cmd(platform, command, job_name)


def _run_platform_run(args):
    _run_platform_run_cmd(args.platform, args.command)


def _run_platform_stop(args) -> None:
    import subprocess

    platform = args.platform
    cfg, _   = _load_platform_config(platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    username  = cfg.get("username", "")
    partition = cfg.get("partition", None)
    job_range = getattr(args, "jobs", None)

    if job_range:
        # ── range mode: cancel job IDs A through B inclusive ──────────
        try:
            a, b = job_range.split("-")
            id_start, id_end = int(a), int(b)
        except ValueError:
            raise ValueError(f"Invalid job range {job_range!r} — expected format A-B")

        job_ids = list(range(id_start, id_end + 1))
        ids_str = " ".join(str(j) for j in job_ids)

        # Preview: query only those specific job IDs for this user
        squeue_cmd = f"squeue -j {','.join(str(j) for j in job_ids)} --format='%.10i %.12P %.30j %.10T %.12M' 2>/dev/null"
        if username:
            squeue_cmd += f" -u {username}"
        result = subprocess.run(["ssh", ssh_host, squeue_cmd], capture_output=True, text=True)
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        n_found = max(0, len(lines) - 1)

        if lines:
            print(result.stdout.strip())
        print(f"\n{n_found} job(s) found in range {id_start}–{id_end} ({len(job_ids)} IDs checked).")

        if not args.yes:
            print(f"Cancel all {len(job_ids)} job IDs in range? [y/N] ", end="", flush=True)
            if input().strip().lower() != "y":
                print("Aborted.")
                return

        scancel_range_cmd = f"scancel {ids_str}"
        if username:
            scancel_range_cmd += f" --user={username}"
        print(f"Running: scancel {id_start}…{id_end}")
        subprocess.run(["ssh", ssh_host, scancel_range_cmd], check=True)
        print(f"Cancelled job IDs {id_start}–{id_end}.")
        return

    # ── default mode: cancel all jobs for user/partition ──────────────
    squeue_cmd = "squeue --format='%.10i %.12P %.30j %.10T %.12M'"
    if username:
        squeue_cmd += f" -u {username}"
    if partition:
        squeue_cmd += f" -p {partition}"

    info = (f"partition={partition}" if partition else "") + \
           (f"  user={username}" if username else "")
    print(f"Querying jobs on {ssh_host}  [{info.strip()}]…")
    result = subprocess.run(["ssh", ssh_host, squeue_cmd], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()

    if len(lines) <= 1:
        print("No running jobs found.")
        return

    print(result.stdout.strip())
    n_jobs = len(lines) - 1  # subtract header

    if not args.yes:
        print(f"\nCancel all {n_jobs} job(s)? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted.")
            return

    scancel_cmd = "scancel"
    if username:
        scancel_cmd += f" -u {username}"
    if partition:
        scancel_cmd += f" -p {partition}"

    print(f"Running: {scancel_cmd}")
    subprocess.run(["ssh", ssh_host, scancel_cmd], check=True)
    print(f"Cancelled {n_jobs} job(s).")


def _run_platform_queue(args):
    import subprocess

    cfg, _ = _load_platform_config(args.platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(f"Platform '{args.platform}' has no ssh host configured")

    username  = cfg.get("username", "")
    partition = getattr(args, "partition", None) or cfg.get("partition", None)
    all_users = getattr(args, "all_users", False)

    # %.Nf = right-align in N chars; %b = gres (gpu:type:count); %m = mem/node
    fmt = "%.10i %.12P %.60j %.10u %.10T %5C %10m %10b %.12M %.12l %R"
    cmd = f"squeue --format={fmt!r}"
    if not all_users and username:
        cmd += f" -u {username}"
    if partition:
        cmd += f" -p {partition}"

    header = f"Queue on {ssh_host}"
    if partition:
        header += f" · partition {partition}"
    if not all_users and username:
        header += f" · user {username}"
    # Partition totals from scontrol (TRES line has exact cpu/mem/gpu counts)
    import re
    sctl_cmd = f"scontrol show partition {partition}" if partition else "scontrol show partition"
    sctl_out = subprocess.run(
        ["ssh", ssh_host, sctl_cmd], capture_output=True, text=True,
    ).stdout
    tres_match = re.search(r"TRES=([^\n]+)", sctl_out)
    total_cpus = total_gpus = mem_gb = 0
    if tres_match:
        tres = tres_match.group(1)
        m = re.search(r"(?<![/\w])cpu=(\d+)", tres)
        if m:
            total_cpus = int(m.group(1))
        m = re.search(r"(?<![/\w])mem=(\d+)([KMGT]?)", tres)
        if m:
            val, unit = int(m.group(1)), m.group(2) or "M"
            mem_gb = val if unit == "G" else val * 1024 if unit == "T" else val // 1024
        m = re.search(r"gres/gpu=(\d+)", tres)
        if m:
            total_gpus = int(m.group(1))

    summary = f"cpu={total_cpus}  mem={mem_gb}GB  gpu={total_gpus}"
    if total_gpus:
        summary += (f"  cpu/gpu={total_cpus / total_gpus:.1f}"
                    f"  mem/gpu={mem_gb / total_gpus:.0f}GB")

    print("=" * 70)
    print(header)
    print("=" * 70)
    subprocess.run(["ssh", ssh_host, cmd], check=True)
    print("=" * 70)
    print(summary)
    print("=" * 70)


def _run_platform(args):
    if args.platform_command == "setup":
        _run_platform_setup(args)
    elif args.platform_command == "status":
        _run_platform_status(args)
    elif args.platform_command == "runi":
        _run_platform_runi(args)
    elif args.platform_command == "setupi":
        _run_platform_setupi(args)
    elif args.platform_command == "run":
        _run_platform_run(args)
    elif args.platform_command == "stop":
        _run_platform_stop(args)
    elif args.platform_command == "queue":
        _run_platform_queue(args)


# ── bench sub-parser ──────────────────────────────────────────────────────────

def _resolve_bench_config(name_or_path: str) -> Path:
    """Resolve a benchmark config name or path to an absolute Path.

    Accepts a full/relative path (used as-is if it exists) or a short name
    resolved against src/configs/eval/ or src/configs/ relative to CWD.
    """
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    stem = name_or_path if not name_or_path.endswith(".yaml") else name_or_path[:-5]
    for subdir in ("src/configs/bench", "src/configs/eval", "src/configs"):
        candidate = Path.cwd() / subdir / f"{stem}.yaml"
        if candidate.exists():
            return candidate
    raise argparse.ArgumentTypeError(
        f"Benchmark config not found: {name_or_path!r}\n"
        f"  Tried: {p.resolve()}\n"
        f"  Tried: {Path.cwd() / 'src/configs/bench' / (stem + '.yaml')}\n"
        f"  Tried: {Path.cwd() / 'src/configs/eval' / (stem + '.yaml')}\n"
        f"  Tried: {Path.cwd() / 'src/configs' / (stem + '.yaml')}"
    )


def _resolve_ablation(name_or_path: str) -> list[Path]:
    """Resolve a comma-separated list of ablation names/paths to resolved Paths.

    Each entry may be a full/relative path (file or directory) or a short name
    resolved against src/configs/ablation/ relative to CWD.
    """
    result = []
    for part in name_or_path.split(","):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        if p.exists():
            result.append(p.resolve())
            continue
        candidate = Path.cwd() / "src" / "configs" / "ablation" / part
        if candidate.exists():
            result.append(candidate)
            continue
        raise argparse.ArgumentTypeError(
            f"Ablation not found: {part!r}\n"
            f"  Tried: {p.resolve()}\n"
            f"  Tried: {candidate}"
        )
    return result


def _ablation_files(ablation: Path) -> list[Path]:
    """Return sorted list of YAML files for a single dir or file."""
    if ablation.is_file():
        return [ablation]
    return sorted(ablation.glob("*.yaml"))


def _ablation_combinations(ablations: list[Path]) -> list[tuple[Path, ...]]:
    """Return the Cartesian product of YAML files across all ablation entries.

    Each entry expands to its set of YAML files (or itself if a single file).
    A single entry with N files → N 1-tuples; two entries with M and N files →
    M*N 2-tuples where both configs are merged for each run.
    """
    import itertools
    per_entry = [_ablation_files(a) for a in ablations]
    return list(itertools.product(*per_entry))


def _repo_rel(path: Path) -> str:
    """Return path relative to CWD (repo root) when possible, else absolute."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _build_bench_parser(sub):
    p = sub.add_parser("bench", help="Benchmark commands")
    bench_sub = p.add_subparsers(dest="bench_command", required=True)

    def _add_bench_args(q):
        q.add_argument(
            "-b", "--benchmark", required=True, type=_resolve_bench_config, metavar="BENCHMARK",
            help="Benchmark config name (resolved from src/configs/bench/) or full path to YAML",
        )
        q.add_argument(
            "-p", "--platform", default=None, metavar="PLATFORM",
            help="Override the platform from the benchmark config's defaults list",
        )
        q.add_argument(
            "-a", "--ablation", default=None, type=_resolve_ablation, metavar="ABLATION",
            help="Comma-separated ablation dirs/files (names resolved from src/configs/ablation/); "
                 "each YAML across all entries is merged on top of the benchmark config and run in sequence",
        )

    p_run = bench_sub.add_parser("run", help="Run benchmark(s) locally")
    _add_bench_args(p_run)

    def _add_fetch_only_args(q):
        q.add_argument(
            "-e", "--entity", default=None, metavar="ENTITY",
            help="W&B entity (user or team). Defaults to the logged-in user's default entity.",
        )
        q.add_argument(
            "-o", "--output", default=None, type=Path, metavar="FILE",
            help="Output CSV path (default: <benchmark>.csv in CWD)",
        )

    def _add_qualit_arg(q):
        q.add_argument(
            "--qualit", action="store_true",
            help="Instead of plotting metrics, look up qualitative images logged to W&B for each "
                 "run in the ablation table and lay them out in a grid.",
        )

    def _add_fetch_args(q):
        _add_bench_args(q)
        _add_fetch_only_args(q)

    p_rrun = bench_sub.add_parser("rrun", help="Submit benchmark(s) as remote jobs via o3b platform run")
    _add_bench_args(p_rrun)
    p_rrun.add_argument(
        "-d", "--deps", default=None, metavar="DEPS",
        help="Comma-separated dep sets that override the platform config's deps field "
             "(e.g. -d diff3f or -d densematcher,diff3f). Controls venv selection.",
    )
    p_rrun.add_argument(
        "--force", action="store_true",
        help="Submit jobs even if they are already running, pending, or recently completed.",
    )
    p_rrun.add_argument(
        "--pull", action="store_true",
        help="Refresh the remote checkout first (`o3b platform setup --pull-only`) so the "
             "jobs run the pushed HEAD. Needed on setup_on_login platforms such as JUPITER, "
             "whose compute nodes cannot reach github and so never pull by themselves. "
             "Aborts without submitting if the pull fails.",
    )
    p_rrun.add_argument(
        "--skip-fetched", action="store_true",
        help="Skip jobs whose ablation combo already has a row in the fetched tables/ CSV.",
    )
    # convenience: run the same invocation as `bench fetch` / `bench viz` instead of
    # submitting, so the identical -b/-a/-p line can be reused for all three steps
    g_rrun = p_rrun.add_mutually_exclusive_group()
    g_rrun.add_argument(
        "--fetch", action="store_true",
        help="Do not submit; behave exactly like `o3b bench fetch` with the same arguments.",
    )
    g_rrun.add_argument(
        "--viz", action="store_true",
        help="Do not submit; behave exactly like `o3b bench viz` with the same arguments.",
    )
    _add_fetch_only_args(p_rrun)
    _add_qualit_arg(p_rrun)

    p_fetch = bench_sub.add_parser("fetch", help="Fetch eval metrics from wandb and save to CSV")
    _add_fetch_args(p_fetch)

    p_viz = bench_sub.add_parser("viz", help="Interactively plot eval metrics from a bench CSV")
    _add_fetch_args(p_viz)
    _add_qualit_arg(p_viz)

    p_wbsync = bench_sub.add_parser(
        "wbsync",
        help="Upload a platform's offline W&B runs from its login node",
    )
    p_wbsync.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_wbsync.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Only print the remote summary of synced/unsynced runs; upload nothing.",
    )
    p_wbsync.add_argument(
        "--clean", action=argparse.BooleanOptionalAction, default=True,
        help="After syncing, delete the local directories of runs that are over — "
             "uploaded, and produced by a SLURM job that is no longer queued "
             "(default). A run whose job is still queued is never deleted, nor is "
             "one that has not been uploaded, nor one whose state cannot be "
             "established; nothing on the server is affected. --no-clean keeps "
             "everything on disk.",
    )
    p_wbsync.add_argument(
        "--clean-old-hours", type=int, default=None, metavar="N",
        help="Extra guard for --clean: additionally spare anything that started "
             "less than N hours ago. Off by default — whether the job is still "
             "queued is the criterion, and it needs no age heuristic. Setting it "
             "also lets non-SLURM runs, whose state cannot otherwise be checked, "
             "become eligible once they are older than N.",
    )
    p_wbsync.add_argument(
        "--dir", default=None, metavar="DIR",
        help="Remote wandb directory (default: <repo>/wandb on the platform).",
    )
    p_wbsync.add_argument(
        "-e", "--entity", default=None, metavar="ENTITY",
        help="W&B entity to upload to. Defaults to whatever the offline runs recorded.",
    )
    p_wbsync.add_argument(
        "--project", default=None, metavar="PROJECT",
        help="W&B project to upload to. Defaults to whatever the offline runs recorded.",
    )


# Classifies (and optionally deletes) the offline run directories on the remote.
# Shipped base64-encoded and run there, so it is always this version rather than
# whatever the cluster checkout happens to hold.
#
# argv: <wandb_dir> <active_job_ids csv> <have_squeue 0|1> <min_age_hours|""> <delete 0|1>
def _b64(text: str) -> str:
    """Shell-quoted base64 of *text*, for shipping a script through ssh intact.

    The alternative -- a heredoc or a `python3 -c` string -- puts the payload
    through bash quoting on a script that is itself arriving on stdin, where a
    stray quote or backslash silently corrupts it.
    """
    import base64
    import shlex

    return shlex.quote(base64.b64encode(text.encode()).decode())


_WBCLEAN_PY = r'''
import datetime, json, os, shutil, sys

wandb_dir, active_csv, have_squeue, min_age, do_delete = sys.argv[1:6]
active = {t for t in active_csv.split(",") if t}
have_squeue = have_squeue == "1"
min_age_h = float(min_age) if min_age else None
do_delete = do_delete == "1"
now = datetime.datetime.now()


def job_id_of(path):
    """The SLURM job that produced this run, or None for a non-SLURM run."""
    try:
        with open(os.path.join(path, "files", "wandb-metadata.json")) as fh:
            jid = (json.load(fh).get("slurm") or {}).get("job_id")
    except Exception:
        return None
    return str(jid) if jid is not None else None


def age_hours(name):
    # offline-run-20260807_171912-gq51znil -> the run's start time
    try:
        stamp = name.split("run-")[1].split("-")[0]
        started = datetime.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
    except Exception:
        return None
    return (now - started).total_seconds() / 3600.0


rows = []
for name in sorted(os.listdir(wandb_dir)):
    path = os.path.join(wandb_dir, name)
    if not name.startswith("offline-run-") or not os.path.isdir(path):
        continue
    synced = os.path.exists(path + ".synced")
    jid = job_id_of(path)
    age = age_hours(name)
    young = min_age_h is not None and age is not None and age < min_age_h

    # Order matters: every branch that cannot *prove* the run is over keeps it.
    if not synced:
        state, keep, why = "pending", True, "not uploaded yet"
    elif jid is not None and jid in active:
        state, keep, why = "running", True, "slurm job %s still queued" % jid
    elif jid is None:
        # No SLURM metadata (a login-node or local run): there is no way to tell
        # whether it is still going, so it only goes on an explicit age rule.
        if min_age_h is not None and not young:
            state, keep, why = "done", False, "no slurm job, older than cutoff"
        else:
            state, keep, why = "unknown", True, "no slurm job id to check"
    elif not have_squeue:
        state, keep, why = "unknown", True, "squeue unavailable, cannot verify"
    elif young:
        state, keep, why = "done", True, "newer than cutoff"
    else:
        state, keep, why = "done", False, "slurm job %s finished" % jid
    rows.append((name, state, keep, why, path))

counts = {}
for _, state, keep, _, _ in rows:
    counts[state] = counts.get(state, 0) + 1
order = ["running", "pending", "done", "unknown"]
print("  " + ", ".join("%s %s" % (counts.get(s, 0), s) for s in order if counts.get(s)))

removable = [r for r in rows if not r[2]]
for name, _state, keep, why, _ in rows:
    if keep:
        print("    KEEP    %-46s %s" % (name, why))
for name, _, _, why, _ in removable:
    print("    %s %-46s %s" % ("DELETE " if do_delete else "would rm", name, why))

if not do_delete:
    sys.exit(0)

freed = 0
for name, _, _, _, path in removable:
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                freed += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)
    marker = path + ".synced"
    if os.path.exists(marker):
        os.remove(marker)
print("  removed %d run(s), freed %.1f MiB" % (len(removable), freed / 1048576.0))
'''


def _run_bench_wbsync(args) -> None:
    """Upload offline W&B runs from a platform's login node.

    Platforms whose compute nodes cannot reach the wandb servers set
    ``wandb_mode: offline`` (JUPITER), so runs accumulate as
    ``<repo>/wandb/offline-run-*`` and never appear in the UI. The login node
    does have connectivity, so the sync runs there over plain ssh — never under
    srun, which would land back on an offline compute node.

    Authentication comes from ``credentials.wandb.api_key`` when the platform
    config sets one, otherwise from whatever ``wandb login`` left in the remote
    ``~/.netrc``.
    """
    import subprocess

    from omegaconf import OmegaConf

    platform = args.platform or "slurm"
    cfg, _ = _load_platform_config(platform)
    if not cfg.get("ssh") or cfg.get("ssh") is False:
        print(f"ERROR: platform '{platform}' has no ssh host — nothing to sync from. "
              f"Offline runs on this machine sync with a plain `wandb sync`.",
              file=sys.stderr)
        raise SystemExit(2)

    (ssh_host, _srun, repo_path, env_path, _path_cuda, path_ws, _hf_cache,
     use_conda, path_conda, _mp_env, modules) = _platform_srun_context(platform)

    wandb_dir = args.dir or (f"{repo_path}/wandb" if repo_path else f"{path_ws}/wandb")
    if Path(wandb_dir).name != "wandb":
        print(f"WARNING: {wandb_dir} is not named 'wandb'; `wandb sync` only "
              f"auto-discovers a directory with that name and will find nothing.")
    mode = str(cfg.get("wandb_mode", "") or "")
    if mode and mode != "offline":
        print(f"NOTE: {platform} runs with wandb_mode={mode!r}, so there may be "
              f"nothing offline to sync.")

    # normally empty: `o3b platform setup` already wrote the key into the
    # remote ~/.netrc, so exporting it here is only a fallback
    api_key = _credential(cfg, "wandb.api_key")

    sync_args = ["--sync-all"]
    if args.entity:
        sync_args += ["--entity", args.entity]
    if args.project:
        sync_args += ["--project", args.project]

    lines = [
        "set -eo pipefail",
        # Lmod's shell functions are not set -e/-u clean, and a non-interactive
        # shell often lacks the init that ~/.bashrc performs (same dance as
        # setup_slurm.sh, which is where these platforms' modules come from).
        *([
            "set +e",
            "if ! command -v module >/dev/null 2>&1 && "
            "[ -f /usr/share/lmod/lmod/init/bash ]; then . /usr/share/lmod/lmod/init/bash; fi",
            *[f'module load {m} || echo "WARNING: module load {m} failed"'
              for m in modules.split()],
            "set -e",
        ] if modules else []),
    ]
    if use_conda:
        lines += [
            f'eval "$("{path_conda}/bin/conda" shell.bash hook)"',
            f"conda activate {env_path}",
        ]
    else:
        lines.append(f"source {env_path}/bin/activate")
    if api_key:
        lines.append(f"export WANDB_API_KEY={api_key}")
    lines += [
        # never inherit the platform's offline setting here: that is exactly what
        # we are undoing, and `wandb sync` under WANDB_MODE=offline is a no-op
        "unset WANDB_MODE",
        f"if [ ! -d {wandb_dir} ]; then "
        f'echo "no wandb directory at {wandb_dir} — nothing to sync"; exit 0; fi',
    ]
    if not args.dry_run:
        lines += [
            # `wandb sync` discovers runs by looking for a ./wandb directory
            # below the cwd, so stand in the *parent*. Standing in wandb_dir
            # itself makes it search wandb/wandb and report "No runs to sync".
            f"cd $(dirname {wandb_dir})",
            f"wandb sync {' '.join(sync_args)}",
        ]

    # ── which runs are over? ──────────────────────────────────────────────────
    # An offline run records the SLURM job that produced it, so "still running"
    # is answered exactly by asking whether that job is still queued -- no age
    # heuristic. squeue must be seen to *succeed*: if it is missing or errors we
    # would read an empty job list as "everything finished" and delete live runs,
    # so the failure is passed through and the classifier keeps everything.
    if args.clean or args.dry_run:
        squeue_user = f" -u {cfg.get('username')}" if cfg.get("username") else ""
        lines += [
            "_active=''; _have_squeue=0",
            "if command -v squeue >/dev/null 2>&1 && "
            f"_sq=$(squeue -h{squeue_user} -o '%i %F' 2>/dev/null); then",
            "    _have_squeue=1",
            "    _active=$(printf '%s' \"${_sq}\" | tr ' ' '\\n' | sed '/^$/d' | "
            "sort -u | paste -sd, -)",
            "fi",
            f"_wbclean=$(mktemp {path_ws or '/tmp'}/.wbclean.XXXXXX.py)",
            f"printf '%s' {_b64(_WBCLEAN_PY)} | base64 -d > \"${{_wbclean}}\"",
            f'python3 "${{_wbclean}}" {wandb_dir} "${{_active}}" "${{_have_squeue}}" '
            f'"{"" if args.clean_old_hours is None else args.clean_old_hours}" '
            f'"{1 if args.clean and not args.dry_run else 0}"',
            'rm -f "${_wbclean}"',
        ]

    if args.dry_run:
        print(f"Listing offline W&B runs on {ssh_host}:{wandb_dir}…")
    else:
        # cleaning is on by default and deletes without prompting, so say so
        cutoff = ("" if args.clean_old_hours is None
                  else f" and older than {args.clean_old_hours}h")
        after = (f"then deleting synced runs whose slurm job has finished{cutoff} "
                 f"(--no-clean to keep)" if args.clean
                 else "keeping every local run directory")
        print(f"Syncing offline W&B runs on {ssh_host}:{wandb_dir}, {after}…")
    proc = subprocess.run(["ssh", ssh_host, "bash -s"],
                          input="\n".join(lines) + "\n", text=True)
    if proc.returncode != 0:
        hint = ("" if api_key else
                "\nHint: no credentials.wandb.api_key in the platform config, so this "
                "relies on the remote ~/.netrc — run `wandb login` on the login node once.")
        print(f"ERROR: wandb sync failed on {ssh_host} (exit {proc.returncode}).{hint}",
              file=sys.stderr)
        raise SystemExit(proc.returncode)


def _run_bench_fetch(args) -> None:
    """Fetch eval/* and train/* metrics from wandb for all job names rrun would submit.

    By default every numeric ``eval/*`` and ``train/*`` key in the run summary
    becomes a column.  A benchmark YAML may narrow that to an explicit list::

        metrics:
          - eval/cam_kpts_trgt_pck01
          - train/time_total_s

    With ``metrics:`` present, only those keys are fetched, in the listed order
    (missing ones are simply left empty for that run).
    """
    import csv
    import re
    import yaml

    bench_stem = args.benchmark.stem

    with open(args.benchmark) as f:
        raw = yaml.safe_load(f) or {}
    metrics_wanted = raw.get("metrics") or None
    if metrics_wanted is not None:
        metrics_wanted = [str(m) for m in metrics_wanted]
    wandb_cfg = raw.get("wandb") or {}
    wb_project = wandb_cfg.get("project", bench_stem)
    wb_entity   = args.entity or wandb_cfg.get("entity", None)
    project_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project

    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]

    try:
        import wandb as _wb_mod
        api = _wb_mod.Api()
    except ImportError:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return

    _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
    def _ablation_col_name(a: Path) -> str:
        try:
            return str(a.relative_to(_ablation_base))
        except ValueError:
            return a.name
    ablation_col_names = [_ablation_col_name(a) for a in args.ablation] if args.ablation else []

    rows = []
    for combo in combos:
        job_name = (
            f"{bench_stem}__{'__'.join(f.stem for f in combo)}"
            if combo else bench_stem
        )
        # wandb run name is f"{timestamp}__{job_name}" where timestamp is MMDD_HHMMSS
        pattern = rf"^\d{{4}}_\d{{6}}__{re.escape(job_name)}$"
        print(f"Querying {project_path!r} for {job_name!r} …")
        try:
            matched = list(api.runs(project_path, filters={"display_name": {"$regex": pattern}}))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        if not matched:
            print("  no runs found")
            continue

        finished = [r for r in matched if r.state == "finished"]
        if not finished:
            print(f"  no finished runs (found {len(matched)} with other states)")
            continue
        run = max(finished, key=lambda r: r.name)

        row: dict = {"job": job_name, "wandb_run": run.name, "state": run.state}
        for col, f in zip(ablation_col_names, combo):
            row[col] = f.stem
        for k, v in run.summary.items():
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if metrics_wanted is not None:
                if k in metrics_wanted:
                    row[k] = round(float(v), 6)
            elif k.startswith(("eval/", "train/")):
                row[k] = round(float(v), 6)
        rows.append(row)
        n_eval  = sum(1 for k in row if k.startswith("eval/"))
        n_train = sum(1 for k in row if k.startswith("train/"))
        print(f"  {run.name}  state={run.state}  "
              f"eval_metrics={n_eval}  train_metrics={n_train}")

    if not rows:
        print("No matching runs found.")
        return

    meta_cols = ["job", "wandb_run", "state"] + ablation_col_names
    if metrics_wanted is not None:
        # keep the YAML order, and keep requested-but-absent keys as empty
        # columns so a typo or a never-logged metric is visible in the table
        metric_cols = list(dict.fromkeys(metrics_wanted))
        missing = [m for m in metric_cols
                   if not any(m in row for row in rows)]
        if missing:
            print(f"WARNING: metrics: requested but not found in any run: "
                  f"{', '.join(missing)}")
    else:
        found = {k for row in rows for k in row}
        metric_cols = (sorted(k for k in found if k.startswith("eval/"))
                       + sorted(k for k in found if k.startswith("train/")))
    fieldnames = meta_cols + metric_cols

    if ablation_col_names:
        safe = [n.replace("/", "_") for n in ablation_col_names]
        table_stem = f"{bench_stem}__{'__'.join(safe)}"
    else:
        table_stem = bench_stem
    output = args.output or Path("tables") / f"{table_stem}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} row(s) → {output}")


def _run_bench_viz_qualit(args) -> None:
    """Load the bench table, look up qualitative images logged to W&B, and grid them.

    Reuses the ablation table built by `o3b bench fetch` (fetching it first if missing)
    instead of re-querying W&B per ablation combo. Only the *first* table entry's run is
    queried to discover which ``qualit/*`` image keys are available; the user picks one or
    more of those keys, and only then are the other runs looked up (one exact `display_name`
    lookup each, cached) to download the first image logged under each selected key. Results
    are tiled: a 2-D grid if two ablation axes vary, a 1-D row if one varies, otherwise a
    near-square grid over all runs.
    """
    import csv
    import math
    import tempfile
    from collections import defaultdict

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import yaml

    bench_stem = args.benchmark.stem
    _ablation_base = Path.cwd() / "src" / "configs" / "ablation"

    def _abl_col(a: Path) -> str:
        try:
            return str(a.relative_to(_ablation_base))
        except ValueError:
            return a.name

    abl_cols = [_abl_col(a) for a in args.ablation] if args.ablation else []
    if abl_cols:
        table_stem = f"{bench_stem}__{'__'.join(c.replace('/', '_') for c in abl_cols)}"
    else:
        table_stem = bench_stem
    csv_path = args.output or Path("tables") / f"{table_stem}.csv"

    if not csv_path.exists():
        print(f"CSV not found at {csv_path} — fetching from wandb first …")
        _run_bench_fetch(args)
    if not csv_path.exists():
        print("ERROR: could not obtain CSV.")
        return

    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # one entry per job: prefer the latest finished run, else just the latest row
    by_job: dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_job[row["job"]].append(row)
    entries = []
    for job in sorted(by_job):
        candidates = [r for r in by_job[job] if r.get("state") == "finished"] or by_job[job]
        entries.append(max(candidates, key=lambda r: r["wandb_run"]))

    if not entries:
        print(f"No rows found in {csv_path}.")
        return

    with open(args.benchmark) as f:
        raw = yaml.safe_load(f) or {}
    wandb_cfg = raw.get("wandb") or {}
    wb_project = wandb_cfg.get("project", bench_stem)
    wb_entity   = args.entity or wandb_cfg.get("entity", None)
    project_path = f"{wb_entity}/{wb_project}" if wb_entity else wb_project

    try:
        import wandb as _wb_mod
        api = _wb_mod.Api()
    except ImportError:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return

    run_by_job: dict[str, object] = {}

    def _get_run(entry):
        """Look up (and cache) the actual W&B Run object for a table entry, by exact name."""
        job = entry["job"]
        if job in run_by_job:
            return run_by_job[job]
        try:
            matched = list(api.runs(project_path, filters={"display_name": entry["wandb_run"]}))
        except Exception as exc:
            print(f"  ERROR fetching run {entry['wandb_run']!r}: {exc}")
            matched = []
        run = matched[0] if matched else None
        if run is None:
            print(f"  WARNING: could not find run {entry['wandb_run']!r} in {project_path!r}")
        run_by_job[job] = run
        return run

    def _get_type(v):
        # summary values that are dicts come back wrapped in a `SummarySubDict`
        # (not a plain dict), so duck-type via `.get()` instead of `isinstance(v, dict)`.
        try:
            return v.get("_type")
        except AttributeError:
            return None

    # ── discover which qualit/* image keys are logged — only the first table entry ──
    first_run = _get_run(entries[0])
    if first_run is None:
        return
    image_keys = sorted(
        k for k, v in first_run.summary.items()
        if k.startswith("qualit/") and _get_type(v) in ("images/separated", "image-file")
    )
    if not image_keys:
        print(f"No qualit/* images found in run {first_run.name!r}'s summary.")
        return

    print("\nAvailable images:")
    for i, k in enumerate(image_keys, 1):
        print(f"  [{i}] {k}")
    while True:
        raw_sel = input(f"\nSelect image(s) [1-{len(image_keys)}] (comma-separated for multiple): ").strip()
        idxs = []
        ok = bool(raw_sel)
        for part in raw_sel.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(image_keys):
                idxs.append(int(part) - 1)
            else:
                ok = False
                break
        if ok:
            selected_keys = [image_keys[i] for i in idxs]
            break
        print(f"  Enter number(s) between 1 and {len(image_keys)}, comma-separated.")

    while True:
        raw_n = input("\nHow many images per key? [1]: ").strip()
        if raw_n == "":
            n_images = 1
            break
        if raw_n.isdigit() and int(raw_n) >= 1:
            n_images = int(raw_n)
            break
        print("  Enter a positive integer.")

    # ── determine which ablation axes actually vary (same rule as the metric viz) ──
    changing_cols = [
        c for c in abl_cols
        if len({e[c] for e in entries if c in e}) > 1
    ]
    fixed_cols = [c for c in abl_cols if c not in changing_cols]
    fixed_desc = ", ".join(
        f"{c}={next(e[c] for e in entries if c in e)}" for c in fixed_cols
    )

    cache_dir = Path(tempfile.gettempdir()) / "o3b_bench_qualit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    history_cache: dict[tuple, list] = {}

    def _all_filenames(run, key: str) -> list:
        """All filenames logged under `key` across the run's full history, in log order.

        `run.summary` only holds the *last* value logged for a key, but each eval batch
        that had qualit enabled logs its own images at its own step — e.g. 5 batches of 4
        samples each show up as 5 separate history rows, 20 images total, not just the last
        batch's 4. We don't sort these: history rows already come back oldest-step-first, and
        each row's own filename list is in original upload order (sample order), so
        concatenating as-is preserves the real ordering.
        """
        cache_key = (run.id, key)
        if cache_key in history_cache:
            return history_cache[cache_key]
        filenames: list = []
        try:
            for row in run.scan_history(keys=[key]):
                v = row.get(key)
                v_type = _get_type(v)
                if v_type == "images/separated":
                    filenames.extend(v.get("filenames") or [])
                elif v_type == "image-file":
                    fname = v.get("path")
                    if fname:
                        filenames.append(fname)
        except Exception as exc:
            print(f"  WARNING: could not read history for {key!r} on run {run.name}: {exc}")
        history_cache[cache_key] = filenames
        return filenames

    def _download_image(run, key: str, idx: int):
        """Download & return a local path to the `idx`-th image logged under `key`, or None."""
        if run is None:
            return None
        filenames = _all_filenames(run, key)
        fname = filenames[idx] if idx < len(filenames) else None
        if not fname:
            return None
        run_cache = cache_dir / run.id
        local_path = run_cache / fname
        if not local_path.exists():
            try:
                run.file(fname).download(root=str(run_cache), replace=False, exist_ok=True)
            except Exception as exc:
                print(f"  WARNING: could not download {fname!r} for run {run.name}: {exc}")
                return None
        return local_path if local_path.exists() else None

    total_pulls = len(selected_keys) * len(entries) * n_images
    progress = {"i": 0}

    def _show_axis(ax, entry, key, img_id):
        if entry is not None:
            progress["i"] += 1
            print(f"  [{progress['i']}/{total_pulls}] {entry['job']}" + " " * 10, end="\r", flush=True)
        run = _get_run(entry) if entry else None
        img_path = _download_image(run, key, img_id) if run is not None else None
        if img_path is not None:
            ax.imshow(mpimg.imread(img_path))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    for key in selected_keys:
        print(f"\nFetching images for {key!r} ({len(entries)} runs × {n_images} image(s) each) …")

        # ── base column/row layout, before replicating per image id ────
        # ── two changing ablation axes → 2-D grid ───────────────────────
        if len(changing_cols) == 2:
            col_a, col_b = changing_cols
            vals_a = sorted({e[col_a] for e in entries if col_a in e})
            vals_b = sorted({e[col_b] for e in entries if col_b in e})
            by_ab = {(e[col_a], e[col_b]): e for e in entries if col_a in e and col_b in e}
            nrows_base, ncols_base = len(vals_a), len(vals_b)
            row_titles, col_titles = vals_a, vals_b
            xlabel, ylabel = col_b, col_a

            def _cell(r, c):
                return by_ab.get((vals_a[r], vals_b[c])), False

        # ── one changing ablation axis → 1-D row (per image id) ─────────
        elif len(changing_cols) == 1:
            col = changing_cols[0]
            vals = sorted({e[col] for e in entries if col in e})
            by_v = {e[col]: e for e in entries if col in e}
            nrows_base, ncols_base = 1, len(vals)
            row_titles, col_titles = None, vals
            xlabel = ylabel = None

            def _cell(r, c):
                return by_v.get(vals[c]), False

        # ── zero or >2 changing axes → near-square grid over all runs ───
        else:
            n = len(entries)
            ncols_base = math.ceil(math.sqrt(n))
            nrows_base = math.ceil(n / ncols_base)
            row_titles, col_titles = None, None
            xlabel = ylabel = None

            def _cell(r, c, _n=n, _ncols=ncols_base):
                idx = r * _ncols + c
                if idx >= _n:
                    return None, True
                e = entries[idx]
                return e, False

        total_nrows = nrows_base * n_images
        fig, axes = plt.subplots(
            total_nrows, ncols_base,
            figsize=(max(4, ncols_base * 3), max(4, total_nrows * 3)),
            squeeze=False,
        )
        for img_id in range(n_images):
            for r in range(nrows_base):
                for c in range(ncols_base):
                    ax = axes[img_id * nrows_base + r][c]
                    entry, is_pad = _cell(r, c)
                    if is_pad:
                        ax.axis("off")
                        continue
                    _show_axis(ax, entry, key, img_id)
                    if img_id == 0 and r == 0 and col_titles is not None:
                        ax.set_title(col_titles[c], fontsize=9)
                    if c == 0:
                        label_parts = []
                        if n_images > 1:
                            label_parts.append(f"img {img_id}")
                        if row_titles is not None:
                            label_parts.append(str(row_titles[r]))
                        if label_parts:
                            ax.set_ylabel("\n".join(label_parts), fontsize=8)

        fig.suptitle(f"{bench_stem}  —  {key}" + (f"\n(fixed: {fixed_desc})" if fixed_desc else ""))
        if xlabel:
            fig.text(0.5, 0.01, xlabel, ha="center", fontsize=9)
        if ylabel:
            fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=9)
        plt.tight_layout()

        print()  # move past the progress line

    plt.show()


def _print_metric_table(header: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a copy-pasteable comma-separated (CSV) table."""
    if title:
        print(f"\n{title}")
    for cells in [header] + rows:
        print(",".join(cells))
    print()


def _run_bench_viz(args) -> None:
    """Load (or fetch) the bench CSV and show an interactive bar-plot for a chosen metric."""
    if getattr(args, "qualit", False):
        _run_bench_viz_qualit(args)
        return

    import csv
    from collections import defaultdict

    bench_stem = args.benchmark.stem
    _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
    if args.ablation:
        abl_names = []
        for a in args.ablation:
            try:
                abl_names.append(str(a.relative_to(_ablation_base)).replace("/", "_"))
            except ValueError:
                abl_names.append(a.name.replace("/", "_"))
        table_stem = f"{bench_stem}__{'__'.join(abl_names)}"
    else:
        table_stem = bench_stem
    csv_path   = args.output or Path("tables") / f"{table_stem}.csv"

    if not csv_path.exists():
        print(f"CSV not found at {csv_path} — fetching from wandb first …")
        _run_bench_fetch(args)

    if not csv_path.exists():
        print("ERROR: could not obtain CSV.")
        return

    with open(csv_path, newline="") as f:
        reader    = csv.DictReader(f)
        all_rows  = list(reader)
        fieldnames = reader.fieldnames or []

    metric_cols = [c for c in fieldnames
                   if c.startswith(("eval/", "train/"))
                   and c not in ("eval/n_samples", "train/epoch")]
    if not metric_cols:
        print("No eval/* or train/* columns found in CSV.")
        return

    # For each job keep only the latest finished run that has at least one metric value
    by_job: dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_job[row["job"]].append(row)

    best_rows = []
    for job in sorted(by_job):
        candidates = [
            r for r in by_job[job]
            if r.get("state") == "finished"
            and any(r.get(m, "").strip() != "" for m in metric_cols)
        ]
        if candidates:
            best_rows.append(max(candidates, key=lambda r: r["wandb_run"]))

    if not best_rows:
        print("No finished runs with metrics found in CSV.")
        return

    available = [m for m in metric_cols if any(r.get(m, "").strip() != "" for r in best_rows)]
    if not available:
        print("No non-empty metric values found.")
        return

    # Interactive metric selection
    print("\nAvailable metrics:")
    for i, m in enumerate(available, 1):
        print(f"  [{i}] {m}")
    while True:
        raw = input(f"\nSelect metric [1-{len(available)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(available):
            metric = available[int(raw) - 1]
            break
        print(f"  Enter a number between 1 and {len(available)}.")

    # Filter to runs that have a value for the chosen metric
    plot_rows = [r for r in best_rows if r.get(metric, "").strip() != ""]
    if not plot_rows:
        print(f"No rows with values for {metric!r}.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    # ── determine which ablation axes actually vary ────────────────────
    # An ablation entry that points at a single YAML file (not a directory)
    # only ever contributes one value, so it's a fixed axis, not a plotted one.
    _abl_base = Path.cwd() / "src" / "configs" / "ablation"
    def _abl_col(a: Path) -> str:
        try:
            return str(a.relative_to(_abl_base))
        except ValueError:
            return a.name

    abl_cols = [_abl_col(a) for a in args.ablation] if args.ablation else []
    changing_cols = [
        c for c in abl_cols
        if len({r[c] for r in plot_rows if c in r}) > 1
    ]
    fixed_cols = [c for c in abl_cols if c not in changing_cols]
    fixed_desc = ", ".join(
        f"{c}={next(r[c] for r in plot_rows if c in r)}" for c in fixed_cols
    )

    # ── two changing ablation axes → 2-D grid heatmap ───────────────────
    if len(changing_cols) == 2:
        col_a, col_b = changing_cols

        vals_a = sorted({r[col_a] for r in plot_rows if col_a in r})
        vals_b = sorted({r[col_b] for r in plot_rows if col_b in r})

        mat = np.full((len(vals_a), len(vals_b)), float("nan"))
        for r in plot_rows:
            if col_a not in r or col_b not in r:
                continue
            ia = vals_a.index(r[col_a])
            ib = vals_b.index(r[col_b])
            mat[ia, ib] = float(r[metric])

        # extend matrix with average row (bottom) and average column (right)
        avg_col  = np.nanmean(mat, axis=1, keepdims=True)   # per row  → right column
        avg_row  = np.nanmean(mat, axis=0, keepdims=True)   # per col  → bottom row
        avg_all  = np.nanmean(mat, keepdims=True).reshape(1, 1)
        ext_mat  = np.block([
            [mat,     avg_col],
            [avg_row, avg_all],
        ])
        xticks = list(vals_b) + ["Avg"]
        yticks = list(vals_a) + ["Avg"]
        na, nb = len(vals_a), len(vals_b)

        table_title = f"{bench_stem}  —  {metric}"
        if fixed_desc:
            table_title += f"  (fixed: {fixed_desc})"
        _print_metric_table(
            header=[f"{col_a} \\ {col_b}"] + xticks,
            rows=[
                [yticks[ia]] + [
                    "" if np.isnan(ext_mat[ia, ib]) else f"{ext_mat[ia, ib]:.4f}"
                    for ib in range(nb + 1)
                ]
                for ia in range(na + 1)
            ],
            title=table_title,
        )

        fig, ax = plt.subplots(figsize=(max(6, (nb + 1) * 0.9 + 1), max(3, (na + 1) * 0.7 + 1)))
        im = ax.imshow(ext_mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label=metric)

        ax.set_xticks(range(nb + 1))
        ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(na + 1))
        ax.set_yticklabels(yticks, fontsize=8)
        ax.set_xlabel(col_b, fontsize=9)
        ax.set_ylabel(col_a, fontsize=9)
        title = f"{bench_stem}  —  {metric}"
        if fixed_desc:
            title += f"\n(fixed: {fixed_desc})"
        ax.set_title(title, fontsize=10)

        # separator lines before the average row/column
        ax.axhline(na - 0.5, color="white", linewidth=1.5, linestyle="--")
        ax.axvline(nb - 0.5, color="white", linewidth=1.5, linestyle="--")

        vmin, vmax = np.nanmin(ext_mat), np.nanmax(ext_mat)
        mid = (vmin + vmax) / 2
        for ia in range(na + 1):
            for ib in range(nb + 1):
                v = ext_mat[ia, ib]
                if not np.isnan(v):
                    color = "white" if v < mid else "black"
                    ax.text(ib, ia, f"{v:.4f}", ha="center", va="center", fontsize=7, color=color)

        plt.tight_layout()
        plt.show()
        return

    # ── single (or zero) changing ablation axis → bar chart ────────────
    if len(changing_cols) == 1:
        col = changing_cols[0]
        labels = [r[col] for r in plot_rows]
    else:
        labels = [r["job"].split("__")[-1] for r in plot_rows]
    values = [float(r[metric]) for r in plot_rows]

    pairs  = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    labels, values = [p[0] for p in pairs], [p[1] for p in pairs]

    avg = sum(values) / len(values)
    labels.append("Average")
    values.append(avg)

    table_title = f"{bench_stem}  —  {metric}"
    if fixed_desc:
        table_title += f"  (fixed: {fixed_desc})"
    _print_metric_table(
        header=[changing_cols[0] if len(changing_cols) == 1 else "job", metric],
        rows=[[lbl, f"{val:.4f}"] for lbl, val in zip(labels, values)],
        title=table_title,
    )

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 5))
    colors  = ["steelblue"] * (len(labels) - 1) + ["darkorange"]
    bars    = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=7)
    bar_title = f"{bench_stem}  —  {metric}"
    if fixed_desc:
        bar_title += f"\n(fixed: {fixed_desc})"
    ax.set_title(bar_title, fontsize=10)
    ax.set_ylabel(metric)
    ax.set_xlabel("category")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()


def _run_bench(args) -> None:
    if args.bench_command == "run":
        _run_bench_run(args)
    elif args.bench_command == "rrun":
        if getattr(args, "fetch", False):
            _run_bench_fetch(args)
        elif getattr(args, "viz", False):
            _run_bench_viz(args)
        else:
            _run_bench_rrun(args)
    elif args.bench_command == "fetch":
        _run_bench_fetch(args)
    elif args.bench_command == "viz":
        _run_bench_viz(args)
    elif args.bench_command == "wbsync":
        _run_bench_wbsync(args)


def _run_bench_run(args) -> None:
    import yaml
    from omegaconf import OmegaConf

    from o3b.dataset.dataset import _load_yaml_with_defaults
    from o3b.dataset.cli import _platform_to_dataset_overrides, _resolve_dataset_config
    from o3b.run import _run_bench_run_with_cfg

    with open(args.benchmark) as f:
        raw = yaml.safe_load(f)

    # ── resolve platform and dataset from defaults list ───────────────────────
    defaults = raw.pop("defaults", []) or []
    default_platform = None
    default_dataset  = None
    for item in defaults:
        if isinstance(item, dict):
            default_platform = item.get("platform", default_platform)
            default_dataset  = item.get("dataset",  default_dataset)

    platform = args.platform if args.platform is not None else (default_platform or "default")

    # ── load base dataset config once (shared across all ablations) ───────────
    overrides = _platform_to_dataset_overrides(platform)
    if default_dataset:
        ds_base = _load_yaml_with_defaults(_resolve_dataset_config(default_dataset), overrides=overrides)
    else:
        ds_base = {}

    # ── inject platform config as 'platform:' so ${platform.path_exps} resolves ─
    try:
        platform_cfg, _ = _load_platform_config(platform)
        platform_cfg_resolved = OmegaConf.to_container(
            OmegaConf.create(platform_cfg), resolve=True
        )
        raw["platform"] = platform_cfg_resolved
    except Exception:
        pass

    # ── collect ablation combinations (Cartesian product across entries) ────────
    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]  # single run with no ablation

    # ── pre-resolve dataset section in isolation so ${key} interpolations
    #    refer to sibling keys (e.g. ${category}) regardless of nesting later ──
    if isinstance(raw.get("dataset"), dict):
        try:
            raw["dataset"] = OmegaConf.to_container(
                OmegaConf.create(raw["dataset"]), resolve=True
            )
        except Exception:
            pass  # leave unresolved; the per-run merge will handle it

    # ── run once per combination ──────────────────────────────────────────────
    for combo in combos:
        if combo:
            # merge all ablation YAMLs in the combo sequentially
            merged_ablation = OmegaConf.create({})
            for ablation_file in combo:
                with open(ablation_file) as f:
                    merged_ablation = OmegaConf.merge(
                        merged_ablation, OmegaConf.create(yaml.safe_load(f) or {})
                    )
            run_raw = OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(dict(raw)), merged_ablation),
                resolve=True,
            )
            combo_stem = "__".join(f.stem for f in combo)
            print(f"\n{'='*60}")
            print(f"Ablation: {combo_stem}")
            print(f"{'='*60}")
        else:
            # resolve the whole raw config together (not just the 'dataset'
            # sub-section) so ${category}-style interpolations that reference
            # sibling top-level keys resolve correctly, mirroring the combo branch
            run_raw = OmegaConf.to_container(
                OmegaConf.create(dict(raw)), resolve=True
            )
            combo_stem = None

        # merge benchmark/ablation dataset section on top of base dataset config
        run_ds = run_raw.get("dataset") or {}
        ds_merged = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(ds_base), OmegaConf.create(run_ds)),
            resolve=True,
        ) if run_ds else dict(ds_base)

        # ── optional training dataset (method.train.dataset_train) ────────────
        # The method creates this dataset lazily by dataset_name (mirroring its
        # pose model); only inject the platform's dataset path overrides here so
        # the named config resolves against the right roots. Keys already set in
        # the ablation win.
        _method_cfg = run_raw.get("method")
        _train_cfg = _method_cfg.get("train") if isinstance(_method_cfg, dict) else None
        _ds_train = _train_cfg.get("dataset_train") if isinstance(_train_cfg, dict) else None
        if isinstance(_ds_train, dict) and _ds_train.get("dataset_name"):
            for _ov in overrides:
                _key, _, _val = _ov.partition("=")
                _ds_train.setdefault(_key, _val)

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        if combo_stem:
            run_name = f"{timestamp}__{args.benchmark.stem}__{combo_stem}"
        else:
            run_name = f"{timestamp}__{args.benchmark.stem}"
        _run_bench_run_with_cfg({**run_raw, "dataset": ds_merged}, run_name)


def _run_bench_sbatch_cmd(platform: str, command: str, job_name: str,
                          deps_override: list | None = None,
                          platform_override: dict | None = None) -> None:
    """Upload a run script + sbatch wrapper and submit via sbatch.

    ``platform_override`` is the ``platform:`` block collected from the run's
    ablation YAMLs (see `_run_bench_rrun`). It is merged over the platform
    config, so an ablation can size its own job — node_count /
    gpu_count_per_node for the multi-GPU ablations, but equally walltime or
    partition — without a platform config per variant. ``deps_override``
    stays separate: it is also settable from the CLI, which wins over both.
    """
    import os, re, subprocess
    from omegaconf import OmegaConf

    cfg, _ = _load_platform_config(platform)
    if platform_override:
        # Hydra composes the platform config in struct mode, which rejects any
        # key the config did not already declare — including a *new* key inside
        # a declared-but-empty mapping, so `env: {}` in slurm.yaml plus an
        # ablation's `platform: {env: {OMP_NUM_THREADS: 5}}` raises
        # "Key 'OMP_NUM_THREADS' is not in struct".  An ablation's platform
        # block is meant to add keys, not only to retune declared ones, so the
        # override is applied with struct off.
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(platform_override)))

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(f"Platform '{platform}' has no ssh host configured")

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    torch_version  = str(cfg.get("torch_version", "2.6.0"))
    deps                 = deps_override if deps_override is not None else list(cfg.get("deps", []) or [])
    deps_tag             = "_".join(sorted(deps)) if deps else ""
    install_flags        = {f"INSTALL_{dep.upper()}": "true" for dep in deps}
    setup          = "true" if cfg.get("setup", False) else "false"
    branch         = str(cfg.get("branch", "main"))
    pull           = str(cfg.get("pull", True)).lower()
    pull_subs      = str(cfg.get("pull_submodules", True)).lower()
    # The compute-node preamble cannot fetch where setup_on_login is set: that
    # flag means the compute nodes have no route to github (which is precisely
    # why setup runs on the login node), and a fetch there fails the whole job.
    # `o3b platform setup` keeps the checkout current instead. Kept separate
    # from the `pull` key itself, which setup on the login node still honours.
    if str(cfg.get("setup_on_login", False)).lower() in ("true", "1", "yes"):
        pull, pull_subs = "false", "false"
    path_home      = cfg.get("path_home", path_ws)
    hf_datasets_cache = cfg.get("path_hf_datasets_cache", "") or ""

    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=local_repo_root,
        ).strip()
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        plain    = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain   # token-free; auth comes from the GITHUB_TOKEN credential helper
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = ""

    repo_path  = f"{path_ws}/{repo_name}" if (path_ws and repo_name) else path_ws
    env_layout = _resolve_env_layout(cfg, repo_path, repo_name)
    env_path   = env_layout["env_path"]

    # empty on platforms with a direct connection (JSC); only LMB needs tfproxy
    _proxy  = cfg.get("http_proxy", "") or ""
    modules = " ".join(str(m) for m in list(cfg.get("modules", []) or []))
    env_vars = {
        "PATH_WS":         path_ws,
        "PATH_CUDA":       path_cuda,
        "PYTHON_VERSION":  python_version,
        "TORCH_VERSION":   torch_version,
        **install_flags,
        **env_layout["env_vars"],
        "DEPS_TAG":              deps_tag,
        "REPO_URL":        repo_url,
        "REPO_NAME":       repo_name,
        "GITHUB_TOKEN":    token,
        "SETUP":           setup,
        "BRANCH":          branch,
        "PULL":            pull,
        "PULL_SUBMODULES": pull_subs,
        "MODULES":         modules,
        **({"TORCH_CUDA_ARCH_LIST": str(cfg.get("torch_cuda_arch_list", "") or "")}
           if cfg.get("torch_cuda_arch_list", "") else {}),
        # wandb cannot reach its servers from a JSC compute node; "offline"
        # needs neither network nor an API key and is synced later from a
        # login node with `wandb sync`.
        **({"WANDB_MODE": str(cfg.get("wandb_mode", "") or "")}
           if cfg.get("wandb_mode", "") else {}),
        **({"TORCH_HOME": str(cfg.get("path_torch_home", "") or "")}
           if cfg.get("path_torch_home", "") else {}),
    }
    if _proxy:
        env_vars.update({
            "HTTP_PROXY": _proxy, "HTTPS_PROXY": _proxy,
            "http_proxy": _proxy, "https_proxy": _proxy,
        })

    from datetime import datetime
    ts = datetime.now().strftime("%m%d_%H%M%S")

    # run script: env preamble (CUDA, venv, cd, setup/pull, distributed) + the
    # actual command.  The command goes through `o3b_launch`, defined by the
    # preamble: it is a plain exec on one GPU and a torchrun launch on more.
    node_count = int(cfg.get("node_count", 1))
    gpu_count  = int(cfg.get("gpu_count_per_node", 1))
    # optional platform key: NCCL_* settings a fabric needs (e.g. P2P/SHM
    # disables for nodes whose peer-to-peer transport is broken)
    _nccl      = OmegaConf.select(cfg, "nccl_env", default=None)
    nccl_env   = ({str(k): str(v)
                   for k, v in OmegaConf.to_container(_nccl, resolve=True).items()}
                  if _nccl else {})
    run_script_content = "\n".join(
        ["#!/usr/bin/env bash", "set -euo pipefail", ""] +
        _srun_env_lines(path_cuda, env_path, repo_path, path_ws, hf_datasets_cache,
                        use_conda=env_layout["use_conda"],
                        path_conda=env_layout["path_conda"],
                        mp_env=_mp_env_from_cfg(cfg),
                        modules=modules,
                        node_count=node_count, gpu_count=gpu_count,
                        nccl_env=nccl_env) +
        ["", f"o3b_launch {command}"]
    )
    remote_run_script = f"{path_ws}/.bench_run_{job_name}_{ts}.sh"

    sbatch_script = _make_sbatch_script(
        cfg,
        job_name=job_name,
        env_vars=env_vars,
        remote_setup_script=remote_run_script,
    )
    remote_sbatch = f"{path_ws}/.bench_sbatch_{job_name}_{ts}.sh"

    # Save both scripts locally with the same timestamp before sending to remote
    local_run    = _save_script_locally(f"bench_run_{job_name}",    run_script_content, ts=ts)
    local_sbatch = _save_script_locally(f"bench_sbatch_{job_name}", sbatch_script,      ts=ts)
    print(f"  saved locally: {local_run}")
    print(f"  saved locally: {local_sbatch}")

    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_run_script} && chmod +x {remote_run_script}"],
        input=run_script_content, text=True, check=True,
    )
    subprocess.run(
        ["ssh", ssh_host, f"cat > {remote_sbatch} && chmod +x {remote_sbatch}"],
        input=sbatch_script, text=True, check=True,
    )

    remote_submit = f"mkdir -p {path_home}/slurm_jobs && sbatch {remote_sbatch}"
    print(f"Submitting sbatch job '{job_name}' on {ssh_host}…")
    subprocess.run(["ssh", ssh_host, remote_submit], check=True)


def _get_existing_jobs_on_platform(platform: str) -> set[str]:
    """Return job names that are pending/running OR completed in the last 24 hours."""
    import subprocess

    try:
        cfg, _ = _load_platform_config(platform)
    except Exception:
        return set()

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        return set()

    username = cfg.get("username", "")

    # pending / running jobs via squeue
    cmd = "squeue --noheader --format=%j"
    if username:
        cmd += f" -u {username}"
    result = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True)
    names: set[str] = set()
    if result.returncode == 0:
        names = {line.strip() for line in result.stdout.splitlines() if line.strip()}

    # completed jobs in the last 24 hours via sacct
    try:
        completed = _fetch_jobs(ssh_host, username, hours=24.0)
        names |= {j["JobName"] for j in completed if j.get("State", "").startswith("COMPLETED")}
    except Exception:
        pass

    return names


def _stage_rel(path: Path) -> str:
    """Repo-relative path usable as a tar arcname (never absolute)."""
    return _repo_rel(path).lstrip("/")


def _upload_local_configs(ssh_host: str, stage_dir: str, files: list[tuple[Path, str]]) -> None:
    """Upload local config files to stage_dir on the remote via a tar pipe."""
    import io
    import subprocess
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for local, rel in files:
            tar.add(local, arcname=rel)
    subprocess.run(
        ["ssh", ssh_host, f"mkdir -p {stage_dir} && tar -xf - -C {stage_dir}"],
        input=buf.getvalue(), check=True,
    )


def _run_bench_rrun(args) -> None:
    """Submit each benchmark/ablation run as a separate sbatch job.

    The local benchmark + ablation YAMLs are uploaded to a staging dir on the
    remote so each job runs with the local (possibly uncommitted) configs
    instead of whatever the remote repo checkout contains after its git pull.
    """
    import shlex
    from datetime import datetime

    from omegaconf import OmegaConf as _OC

    platform = args.platform or "slurm"
    bench_stem = args.benchmark.stem
    cli_deps = [d.strip() for d in args.deps.split(",")] if getattr(args, "deps", None) else None

    platform_cfg, _ = _load_platform_config(platform)
    ssh_host = platform_cfg.get("ssh")
    path_ws = platform_cfg.get("path_ws", "")
    upload_configs = bool(ssh_host) and ssh_host is not False and bool(path_ws)
    if not upload_configs:
        print("WARNING: platform has no ssh host / path_ws — remote jobs will use "
              "the repo-checkout configs on the remote, not the local ones")

    # before anything is submitted, so a failed pull queues nothing
    _maybe_pull_platform(args, platform)

    if args.ablation:
        combos = _ablation_combinations(args.ablation)
        if not combos:
            print(f"WARNING: no YAML files found in {args.ablation!r}")
            return
    else:
        combos = [()]

    force = getattr(args, "force", False)
    n_total = len(combos)
    print(f"Checking {n_total} job(s) on {platform}…")
    existing_jobs = set() if force else _get_existing_jobs_on_platform(platform)

    # build set of already-fetched ablation combos from the tables/ CSV
    fetched_combos: set[tuple] = set()
    if getattr(args, "skip_fetched", False):
        import csv as _csv
        _ablation_base = Path.cwd() / "src" / "configs" / "ablation"
        def _abl_col(a: Path) -> str:
            try:
                return str(a.relative_to(_ablation_base))
            except ValueError:
                return a.name
        abl_col_names = [_abl_col(a) for a in args.ablation] if args.ablation else []
        if abl_col_names:
            safe = [n.replace("/", "_") for n in abl_col_names]
            table_stem = f"{bench_stem}__{'__'.join(safe)}"
        else:
            table_stem = bench_stem
        table_path = Path("tables") / f"{table_stem}.csv"
        if table_path.exists():
            with open(table_path, newline="") as _f:
                for _row in _csv.DictReader(_f):
                    fetched_combos.add(tuple(_row.get(col, "") for col in abl_col_names))
            print(f"Loaded {len(fetched_combos)} fetched combo(s) from {table_path}")
        else:
            print(f"WARNING: --skip-fetched set but table not found: {table_path}")

    n_existing = 0
    n_submitted = 0
    width = len(str(n_total))

    for i, combo in enumerate(combos, 1):
        # collect the ablation YAMLs' platform: block (merged in combo order,
        # later files win) — deps are unioned instead, and job sizing keys such
        # as node_count / gpu_count_per_node override the platform config
        ablation_deps: list | None = None
        ablation_platform = _OC.create({})
        for f in combo:
            try:
                acfg = _OC.load(f)
                file_platform = _OC.select(acfg, "platform", default=None)
                if file_platform is not None:
                    ablation_platform = _OC.merge(ablation_platform, file_platform)
                file_deps = _OC.select(acfg, "platform.deps", default=None)
                if file_deps is not None:
                    if ablation_deps is None:
                        ablation_deps = []
                    ablation_deps.extend(list(file_deps))
            except Exception:
                pass
        platform_override = _OC.to_container(ablation_platform, resolve=False)
        platform_override.pop("deps", None)  # handled by deps_override
        # precedence: CLI -d > ablation platform.deps > platform config deps
        effective_deps = cli_deps if cli_deps is not None else ablation_deps

        if combo:
            job_name = f"{bench_stem}__{'__'.join(f.stem for f in combo)}"
        else:
            job_name = bench_stem

        prefix = f"[{i:{width}}/{n_total}]"
        combo_key = tuple(f.stem for f in combo)
        if fetched_combos and combo_key in fetched_combos:
            n_existing += 1
            print(f"{prefix} skip (in table) {job_name}")
            continue
        if job_name in existing_jobs:
            n_existing += 1
            print(f"{prefix} skip   {job_name}")
            continue

        if upload_configs:
            # stage the local YAMLs outside the remote repo (so the job's git
            # pull stays clean) and point the remote command at the staged copies
            ts = datetime.now().strftime("%m%d_%H%M%S")
            stage = f"{path_ws}/.bench_cfgs/{job_name}_{ts}"
            files = [(args.benchmark, _stage_rel(args.benchmark))]
            files += [(f, _stage_rel(f)) for f in combo]
            _upload_local_configs(ssh_host, stage, files)
            bench_arg = f"{stage}/{_stage_rel(args.benchmark)}"
            abl_args = [f"{stage}/{_stage_rel(f)}" for f in combo]
        else:
            bench_arg = _repo_rel(args.benchmark)
            abl_args = [_repo_rel(f) for f in combo]

        parts = ["o3b", "bench", "run", "-b", bench_arg, "-p", platform]
        if combo:
            # pass each file individually; the remote `bench run` will receive a
            # comma-joined -a arg so its own outer-product logic runs the same combo
            parts += ["-a", ",".join(abl_args)]
        remote_cmd = " ".join(shlex.quote(p) for p in parts)

        _size = " ".join(f"{k}={platform_override[k]}"
                         for k in ("node_count", "gpu_count_per_node")
                         if k in platform_override)
        print(f"{prefix} submit {job_name}"
              + (f" [{_size}]" if _size else "")
              + (f" (local configs → {stage})" if upload_configs else ""))
        _run_bench_sbatch_cmd(platform, remote_cmd, job_name,
                              deps_override=effective_deps,
                              platform_override=platform_override)
        n_submitted += 1
        time.sleep(1)

    print(f"\nDone — {n_submitted} submitted, {n_existing} already running/pending/completed"
          + (f", {n_total - n_submitted - n_existing} skipped for other reasons" if n_submitted + n_existing < n_total else ""))


# ── main ──────────────────────────────────────────────────────────────────────

def _normalise_ablation_argv(argv) -> list[str]:
    """Rewrite ``-a <value>`` / ``-a=<value>`` into ``--ablation=<value>``.

    Dataset ablations are fragments of CLI arguments (``-c backpack``), and
    argparse refuses an option value that starts with '-' unless it is attached
    with '='.
    """
    argv = list(argv)
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-a", "--ablation") and i + 1 < len(argv):
            out.append(f"--ablation={argv[i + 1]}")
            i += 2
            continue
        if tok.startswith("-a=") or tok.startswith("--ablation="):
            out.append(f"--ablation={tok.partition('=')[2]}")
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="o3b",
        description="o3b CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_dataset_parser(sub)
    _build_bench_parser(sub)
    _build_platform_parser(sub)

    argv = _normalise_ablation_argv(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    if args.command == "dataset":
        _run_dataset(args, parser=parser, argv=argv)
    elif args.command == "bench":
        _run_bench(args)
    elif args.command == "platform":
        _run_platform(args)


if __name__ == "__main__":
    main()
