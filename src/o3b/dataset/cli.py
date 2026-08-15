"""
o3b dataset CLI.

Usage:
  od3d_dataset fetch  -d dm_object          [--url URL] [--platform PLATFORM]
  od3d_dataset index  -d dm_object          [--db index.db] [--platform PLATFORM]
  od3d_dataset init   -d dm_object          [--limit N] [--override] [--platform PLATFORM]
  od3d_dataset viz    -d dm_object_pair     [--db index.db] [--limit N] [--object-id ID] [--render] [--platform PLATFORM]
  od3d_dataset sshfs  -d uco3d -p slurm     [--unmount] [--dry-run]
  od3d_dataset cp-tform-obj-type -d every9d_v3 -t can_pose_v5 [-p slurm] [--dry-run]
  od3d_dataset select-subset -d every9d_v3 -s every9d_v3_test [--objects N] [--views N]

Pair datasets (*_object_pair, *_frame_object_pair) need no separate index
step — pairs are derived at load time from the base index (index.db / frames.db).

-d / --config accepts either a short name (e.g. dm_object_pair, resolved from
configs/dataset/) or a full path to a YAML file.  The YAML must contain a 'class_name'
field that matches a registered dataset (e.g. 'DenseMatcher').
When --platform is given the platform's path_datasets_raw / path_datasets_preprocess
values override the corresponding variables in the dataset config before Hydra
resolves any ${} interpolations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from o3b.dataset.dataset import DatasetConfig, _REGISTRY_DATASETS, _ensure_dataset_imported
from o3b.dataset.grid import GRID_COLS, GRID_OBJECTS, GRID_VIEWS, PREFETCH


def _resolve_dataset_config(name_or_path: str) -> Path:
    """Resolve a dataset config name or path to an absolute Path.

    Accepts either a full path (used as-is if it exists) or a short name like
    'dm_object_pair' which is looked up in configs/dataset/.
    """
    p = Path(name_or_path)
    if p.exists():
        return p
    configs_dir = (Path(__file__).parent.parent.parent / "configs" / "dataset").resolve()
    stem = name_or_path if not name_or_path.endswith(".yaml") else name_or_path[:-5]
    candidate = configs_dir / f"{stem}.yaml"
    if candidate.exists():
        return candidate
    raise argparse.ArgumentTypeError(
        f"Dataset config not found: {name_or_path!r}\n"
        f"  Tried path: {p.resolve()}\n"
        f"  Tried name: {candidate}"
    )


def _platform_to_dataset_overrides(platform: str) -> list[str]:
    """Return Hydra override strings derived from the platform config."""
    from o3b.cli import _load_platform_config
    plat_cfg, _ = _load_platform_config(platform)
    overrides: list[str] = []
    if raw := plat_cfg.get("path_datasets_raw"):
        overrides.append(f"path_datasets_raw={raw}")
    if pre := plat_cfg.get("path_datasets_preprocess"):
        overrides.append(f"path_datasets_preprocess={pre}")
    # Where datasets that live on another machine are reachable from *this*
    # platform: the real directory on the cluster, an sshfs mount point locally.
    if sshfs := plat_cfg.get("path_datasets_sshfs"):
        overrides.append(f"path_datasets_sshfs={sshfs}")
    return overrides


def _load_class_from_config(config_path: Path, overrides: list[str] | None = None):
    cfg = DatasetConfig.from_yaml(config_path, overrides=overrides)
    _ensure_dataset_imported(cfg.class_name)

    lower = cfg.class_name.lower()
    for key, cls in _REGISTRY_DATASETS.items():
        if key.lower() == lower:
            return cls, cfg

    known = sorted(_REGISTRY_DATASETS)
    print(
        f"ERROR: class_name '{cfg.class_name}' from {config_path} is not registered.\n"
        f"Known datasets: {known if known else '(none — is the subpackage installed?)'}",
        file=sys.stderr,
    )
    sys.exit(1)


def _run_sshfs(args) -> None:
    """`o3b dataset sshfs -d <dataset> -p <platform>`.

    Sources come from the dataset config resolved *with* the platform's paths
    (where the data really is), targets from the same config resolved without
    them (where this machine expects to find it).  A platform with no ssh host
    is a local one and there is nothing to mount.
    """
    from o3b.cli import _load_platform_config
    from o3b.dataset.sshfs import mount

    if args.platform == "default":
        print("sshfs needs a remote platform, e.g. -p slurm.", file=sys.stderr)
        sys.exit(1)

    plat_cfg, _ = _load_platform_config(args.platform)
    host = plat_cfg.get("ssh")
    if not host or host is True:
        print(f"Platform '{args.platform}' defines no ssh host to mount from.", file=sys.stderr)
        sys.exit(1)

    cfg_local  = DatasetConfig.from_yaml(args.config, overrides=_platform_to_dataset_overrides("default"))
    cfg_remote = DatasetConfig.from_yaml(args.config, overrides=_platform_to_dataset_overrides(args.platform))

    verb = "Unmounting" if args.unmount else "Mounting"
    print(f"{verb} {args.config.stem} from {host} ({args.platform}):")
    mount(cfg_local, cfg_remote, host, unmount=args.unmount, dry_run=args.dry_run,
          read_only=getattr(args, "read_only", False))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="od3d_dataset",
        description="od3d dataset CLI — fetch, index, and visualise any registered dataset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_config(p):
        p.add_argument(
            "-d", "--config", required=True, type=_resolve_dataset_config, metavar="DATASET",
            help="Dataset config name (e.g. dm_object_pair, resolved from "
                 "configs/dataset/) or full path to a YAML file",
        )
        p.add_argument(
            "--platform", default="default", metavar="PLATFORM",
            help="Platform name whose path_datasets_raw / path_datasets_preprocess "
                 "override the dataset config paths (default: default)",
        )
        p.add_argument(
            "-c", "--categories", default=None, metavar="CATEGORIES",
            help="Comma-separated categories overriding the config's 'categories' field "
                 "(e.g. -c backpack or -c backpack,book)",
        )

    p_fetch = sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL", help="ZIP download URL")

    p_index = sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite output file (default: <path_preprocess>/index.db)",
    )

    p_init = sub.add_parser(
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

    p_tform = sub.add_parser(
        "tform",
        help="Interactive axis-convention viewer — determine obj_gl_tform4x4_obj_raw for the dataset",
    )
    _add_config(p_tform)
    p_tform.add_argument(
        "--limit", type=int, default=20, metavar="N",
        help="Max objects to browse (default: 20)",
    )

    p_sshfs = sub.add_parser(
        "sshfs",
        help="Mount the dataset's directories from a remote platform over sshfs",
    )
    _add_config(p_sshfs)
    p_sshfs.add_argument("--unmount", "-u", action="store_true",
                         help="Unmount instead of mounting")
    p_sshfs.add_argument("--dry-run", action="store_true",
                         help="Print the sshfs/fusermount commands without running them")
    p_sshfs.add_argument("--read-only", action="store_true",
                         help="Mount read-only (blocks 'index'/'init' writing their caches remotely)")

    p_cptf = sub.add_parser(
        "cp-tform-obj-type",
        help="Copy this dataset's tform_obj_type directory to a new one under "
             "<path_preprocess>/tform_obj/",
    )
    _add_config(p_cptf)
    p_cptf.add_argument("-t", "--target", required=True, metavar="NAME",
                        help="Name of the new tform_obj_type subdirectory, e.g. can_pose_v5")
    p_cptf.add_argument("-n", "--dry-run", action="store_true",
                        help="Print the rsync that would run, without copying")
    p_cptf.add_argument("--override", action="store_true",
                        help="Copy into an existing target instead of refusing, and "
                             "rewrite its dataset config")
    p_cptf.add_argument("--no-config", action="store_true",
                        help="Only copy the directory; do not write configs/dataset/<target>.yaml")

    p_todb = sub.add_parser(
        "tform-obj-to-db",
        help="Convert tform_obj/<type>/ into a single tform_obj/<type>.db",
    )
    _add_config(p_todb)
    p_todb.add_argument("--override", action="store_true",
                        help="Rebuild the .db even if it already exists")
    p_todb.add_argument("-j", "--workers", type=int, default=32, metavar="N",
                        help="Threads reading the per-sequence files (default: 32)")

    p_axes = sub.add_parser(
        "axes-tform-obj-type",
        help="Per-category axis editor: view a category's objects in their canonical "
             "frame, swap/flip the axes for the whole category, and write the result "
             "back to its tform_obj store",
    )
    _add_config(p_axes)
    p_axes.add_argument(
        "-r", "--reference", default=None, metavar="DATASET",
        help="Second dataset config shown alongside for comparison (read-only), "
             "e.g. -r every9d_v3",
    )
    p_axes.add_argument("--objects", type=int, default=GRID_OBJECTS, metavar="N",
                        help="Objects shown per category (default: 16)")
    p_axes.add_argument("--views", type=int, default=GRID_VIEWS, metavar="N",
                        help="Views shown per object, stacked downwards "
                             "(default: 3)")
    p_axes.add_argument("--cols", type=int, default=GRID_COLS, metavar="N",
                        help="Objects side by side before a new block of rows "
                             "starts (default: 8)")
    p_axes.add_argument("--prefetch", type=int, default=PREFETCH, metavar="N",
                        help="Warm the crop cache for the next N categories in the "
                             "background, 0 to disable (default: %(default)s)")
    p_axes.add_argument("--category", default=None, metavar="NAME",
                        help="Start on this category")

    p_sax = sub.add_parser(
        "select-and-axes-tform-obj-type",
        help="Select a category's objects, then swap/flip the axes of those "
             "objects alone and write the result back to its tform_obj store — "
             "select-subset and axes-tform-obj-type in one window (Tab switches)",
    )
    _add_config(p_sax)
    p_sax.add_argument("-s", "--subset", default=None, metavar="NAME",
                       help="Also keep the selection as "
                            "<path_preprocess>/subset/<NAME>.yaml, loaded at start "
                            "and autosaved (default: the selection only aims the "
                            "rotation and is not written anywhere)")
    p_sax.add_argument("--objects", type=int, default=GRID_OBJECTS, metavar="N",
                       help="Objects shown per page (default: 16)")
    p_sax.add_argument("--views", type=int, default=GRID_VIEWS, metavar="N",
                       help="Views shown per object, stacked downwards (default: 3)")
    p_sax.add_argument("--cols", type=int, default=GRID_COLS, metavar="N",
                       help="Objects side by side before a new block of rows "
                            "starts (default: 8)")
    p_sax.add_argument("--prefetch", type=int, default=PREFETCH, metavar="N",
                       help="Warm the crop cache for the first page of the next N "
                            "categories in the background, 0 to disable "
                            "(default: %(default)s)")
    p_sax.add_argument("--category", default=None, metavar="NAME",
                       help="Start on this category")
    p_sax.add_argument("--max-count", type=int, default=None, metavar="N",
                       help="Offer at most N objects per category, which is what "
                            "makes opening a big one fast (the walk stops there). "
                            "Everything already selected is offered on top of the "
                            "cap (default: the whole category)")
    p_sax.add_argument("--max-height", type=int, default=None, metavar="PX",
                       help="Scale the grid to at most this many pixels tall "
                            "(default: the screen height)")

    p_sel = sub.add_parser(
        "select-subset",
        help="Pick the objects of a named subset: page through a category's objects "
             "and toggle each one in or out, writing "
             "<path_preprocess>/subset/<name>.yaml",
    )
    _add_config(p_sel)
    p_sel.add_argument("-s", "--subset", required=True, metavar="NAME",
                       help="Subset name, e.g. every9d_v3_test — stored as "
                            "<path_preprocess>/subset/<NAME>.yaml and used by a "
                            "dataset config's subset_name field")
    p_sel.add_argument("--objects", type=int, default=GRID_OBJECTS, metavar="N",
                       help="Objects shown per page (default: 16)")
    p_sel.add_argument("--views", type=int, default=GRID_VIEWS, metavar="N",
                       help="Views shown per object, stacked downwards "
                            "(default: 3)")
    p_sel.add_argument("--cols", type=int, default=GRID_COLS, metavar="N",
                       help="Objects side by side before a new block of rows "
                            "starts (default: 8)")
    p_sel.add_argument("--prefetch", type=int, default=PREFETCH, metavar="N",
                       help="Warm the crop cache for the first page of the next N "
                            "categories in the background, 0 to disable "
                            "(default: %(default)s)")
    p_sel.add_argument("--category", default=None, metavar="NAME",
                       help="Start on this category")
    p_sel.add_argument("--max-count", type=int, default=None, metavar="N",
                       help="Offer at most N objects per category, which is what "
                            "makes opening a big one fast (the walk stops there). "
                            "Everything already selected is offered on top of the "
                            "cap (default: the whole category)")
    p_sel.add_argument("--init-count", type=int, default=None, metavar="N",
                       help="Offer only the categories that start with exactly N "
                            "objects selected — reviewing a slice of the standing "
                            "selection instead of filling one up (default: all "
                            "categories)")
    p_sel.add_argument("--target-count", type=int, default=None, metavar="N",
                       help="Objects wanted per category: categories already at N "
                            "are reported and skipped, so only the ones still short "
                            "of it are offered (default: no target, all categories)")
    p_sel.add_argument("--max-height", type=int, default=None, metavar="PX",
                       help="Scale the grid to at most this many pixels tall "
                            "(default: the screen height)")

    p_vis = sub.add_parser("viz", help="Show dataset summary and optionally render meshes")
    _add_config(p_vis)
    p_vis.add_argument(
        "--db", type=Path, default=None, metavar="FILE",
        help="SQLite index file (default: <path_preprocess>/index.db)",
    )
    p_vis.add_argument("--limit", type=int, default=20, metavar="N",
                       help="Max objects to show (default: 20)")
    p_vis.add_argument("--object-id", default=None, metavar="ID",
                       help="Show only this object")
    p_vis.add_argument("--filter-has-kpts", action="store_true",
                       help="Only show objects that have obj_kpts3d in the index")
    p_vis.add_argument("--render", action="store_true",
                       help="Open interactive viser viewer")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N",
                       help="Number of viewpoints to render when --render is set (default: 4)")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender",
                       help="Renderer backend for --render-frames (default: pyrender)")
    p_vis.add_argument("--debug", action="store_true",
                       help="Show front/top/right camera frustums in the viser scene")
    p_vis.add_argument("--object-centric", action="store_true",
                       help="Object-centric view: place object at world origin, camera in object space")

    args = parser.parse_args(argv)
    from o3b.cli import _categories_to_dataset_overrides

    if args.command == "sshfs":
        _run_sshfs(args)
        return

    if args.command == "tform-obj-to-db":
        from o3b.dataset.copy_tform_obj import run_tform_obj_to_db
        run_tform_obj_to_db(args.config, platform=args.platform,
                            override=args.override, workers=args.workers)
        return

    if args.command == "cp-tform-obj-type":
        from o3b.dataset.copy_tform_obj import copy_tform_obj_type
        copy_tform_obj_type(args.config, args.target, platform=args.platform,
                            dry_run=args.dry_run, override=args.override,
                            write_config=not args.no_config)
        return

    overrides = (_platform_to_dataset_overrides(args.platform)
                 + _categories_to_dataset_overrides(args.categories))
    cls, cfg = _load_class_from_config(args.config, overrides=overrides)

    if args.command == "axes-tform-obj-type":
        from o3b.dataset.axes_tform_obj import run_axes_editor
        ref_cls = ref_cfg = None
        if args.reference:
            ref_cls, ref_cfg = _load_class_from_config(
                _resolve_dataset_config(args.reference), overrides=overrides)
        run_axes_editor(cls, cfg, reference=ref_cfg, ref_cls=ref_cls,
                        n_objects=args.objects, n_views=args.views,
                        cols=args.cols, prefetch=args.prefetch,
                        category=args.category)
    elif args.command == "select-and-axes-tform-obj-type":
        from o3b.dataset.select_axes import run_select_axes_editor
        run_select_axes_editor(cls, cfg, args.subset, dataset_name=args.config.stem,
                               n_objects=args.objects, n_views=args.views,
                               cols=args.cols, prefetch=args.prefetch,
                               category=args.category, max_height=args.max_height,
                               max_count=args.max_count)
    elif args.command == "select-subset":
        from o3b.dataset.select_subset import run_subset_editor
        run_subset_editor(cls, cfg, args.subset, dataset_name=args.config.stem,
                          n_objects=args.objects, n_views=args.views,
                          cols=args.cols, prefetch=args.prefetch,
                          category=args.category, max_height=args.max_height,
                          target_count=args.target_count,
                          init_count=args.init_count,
                          max_count=args.max_count)
    elif args.command == "tform":
        from o3b.dataset.tform import run_tform_viewer
        run_tform_viewer(cls, cfg, limit=args.limit)
    elif args.command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.command == "index":
        cls.index(cfg, db=args.db)
    elif args.command == "init":
        cls.init(cfg, limit=args.limit, override=args.override)
    elif args.command == "viz":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
            debug=args.debug,
            obj_centric=args.object_centric,
        )


if __name__ == "__main__":
    main()
