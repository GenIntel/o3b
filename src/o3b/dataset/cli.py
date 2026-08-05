"""
o3b dataset CLI.

Usage:
  od3d_dataset fetch  -d hc3d_object        [--url URL] [--platform PLATFORM]
  od3d_dataset index  -d hc3d_object        [--db index.db] [--platform PLATFORM]
  od3d_dataset init   -d hc3d_object        [--limit N] [--override] [--platform PLATFORM]
  od3d_dataset viz    -d hc3d_object_pair   [--db index.db] [--limit N] [--object-id ID] [--render] [--platform PLATFORM]

Pair datasets (hc3d_object_pair, hc3d_frame_object_pair) need no separate index
step — pairs are derived at load time from the base index (index.db / frames.db).

-d / --config accepts either a short name (e.g. housecorr3d_object_pair, resolved from
configs/dataset/) or a full path to a YAML file.  The YAML must contain a 'class_name'
field that matches a registered dataset (e.g. 'HouseCorr3D').
When --platform is given the platform's path_datasets_raw / path_datasets_preprocess
values override the corresponding variables in the dataset config before Hydra
resolves any ${} interpolations.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from o3b.dataset.dataset import DatasetConfig, _REGISTRY_DATASETS, _ensure_dataset_imported


def _resolve_dataset_config(name_or_path: str) -> Path:
    """Resolve a dataset config name or path to an absolute Path.

    Accepts either a full path (used as-is if it exists) or a short name like
    'housecorr3d_object_pair' which is looked up in configs/dataset/.
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="od3d_dataset",
        description="od3d dataset CLI — fetch, index, and visualise any registered dataset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_config(p):
        p.add_argument(
            "-d", "--config", required=True, type=_resolve_dataset_config, metavar="DATASET",
            help="Dataset config name (e.g. housecorr3d_object_pair, resolved from "
                 "configs/dataset/) or full path to a YAML file",
        )
        p.add_argument(
            "--platform", default="default", metavar="PLATFORM",
            help="Platform name whose path_datasets_raw / path_datasets_preprocess "
                 "override the dataset config paths (default: default)",
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
    overrides = _platform_to_dataset_overrides(args.platform)
    cls, cfg = _load_class_from_config(args.config, overrides=overrides)

    if args.command == "tform":
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
