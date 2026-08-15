"""OpenTT – Extended OpenTTGames dataset.

Annotations (JSON) are fetched from:
  https://github.com/moamal01/table_tennis_data

Videos (mp4) are downloaded from:
  https://lab.osai.ai/datasets/openttgames/data/

Each JSON file maps frame numbers to event labels, e.g.:
  {"44": "bounce", "58": "left_forehand_lob right_leaning both_feet_planted", "98": "net"}

Events fall into three categories:
  - ball events   : "bounce", "net"
  - point events  : "left_net", "right_winner", "left_double_bounce", etc.
  - stroke events : "<side>_<hand>_<technique> <lean> <feet>"  (space-separated triple)
  - empty         : "empty_event"
Frames without an annotation receive event=None.

Directory layout after fetch():
  {path_raw}/
      annotations/
          train/  game_1.json … game_5.json
                  game_1_ball.json … game_5_ball.json
          test/   test_1.json … test_7.json
                  test_1_ball.json … test_7_ball.json
      videos/
          train/  game_1.mp4 … game_5.mp4
          test/   test_1.mp4 … test_7.mp4

After index():
  {path_preprocess}/
      manifest.json    # {video_name: n_frames} — speeds up repeated _setup() calls

DatasetConfig.extra knobs:
  frame_stride  int   step between consecutive sampled frames within a clip  (default 1)
  clip_stride   int   step between clip start frames                         (default scene_length * frame_stride)
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import torch

from o3b.data.datatypes.scene import Scene, SceneBatch, collate_scenes
from o3b.dataset.dataset import (
    ConfigurableDataset,
    DatasetConfig,
    register_dataset,
)

# ── constants ─────────────────────────────────────────────────────────────────

_TRAIN_NAMES = [f"game_{i}" for i in range(1, 6)]
_TEST_NAMES  = [f"test_{i}"  for i in range(1, 8)]

_VIDEO_BASE = "https://lab.osai.ai/datasets/openttgames/data"
_ANNO_RAW   = (
    "https://raw.githubusercontent.com/moamal01/table_tennis_data/main"
    "/data/raw/game_data"
)


# ── dataset class ─────────────────────────────────────────────────────────────

@register_dataset("OpenTT")
class OpenTT(ConfigurableDataset):
    """Sliding-window scene dataset built from the Extended OpenTTGames videos.

    Each item is a Scene containing:
      rgbs   (T, H, W, 3) float32 in [0, 1]
      events  list of T entries: str event label or None
    """

    # ── path helpers ───────────────────────────────────────────────────────────

    @property
    def _path_raw(self) -> Path:
        return self.cfg.path_raw or self.cfg.root / "opentt"

    @property
    def _path_preprocess(self) -> Path:
        return self.cfg.path_preprocess or self._path_raw

    @property
    def _path_annotations(self) -> Path:
        return self._path_raw / "annotations"

    @property
    def _path_videos(self) -> Path:
        return self._path_raw / "videos"

    @property
    def _manifest_path(self) -> Path:
        return self._path_preprocess / "manifest.json"

    # ── setup ──────────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._annotations: dict[str, dict[int, str]] = {}
        self._balls: dict[str, dict[int, Optional[tuple]]] = {}
        self._clips: list[dict] = []

        frame_stride: int = int(self.cfg.extra.get("frame_stride", 1))
        scene_len:    int = self.cfg.scene_length
        clip_stride:  int = int(self.cfg.extra.get("clip_stride", scene_len * frame_stride))

        # Load manifest (written by index()) if available; avoids re-opening every video.
        manifest: dict[str, int] = {}
        if self._manifest_path.exists():
            with self._manifest_path.open() as fh:
                manifest = json.load(fh)

        split = self.cfg.split
        if split in ("train", "all"):
            self._ingest_split("train", _TRAIN_NAMES, frame_stride, scene_len, clip_stride, manifest)
        if split in ("test", "all"):
            self._ingest_split("test",  _TEST_NAMES,  frame_stride, scene_len, clip_stride, manifest)

        # subset/<subset_name>.yaml selects clips by hand.  OpenTT has no object
        # id, so an entry is either a video name (the whole video's clips) or one
        # clip's scene_id, <name>_<first frame>.  Before the cap, so the cap
        # counts what the selection left.
        subset = self.subset()
        if subset is not None:
            before = len(self._clips)
            self._clips = [
                c for c in self._clips
                if subset.has_item(c["name"], f"{c['name']}_{c['indices'][0]:07d}")
            ]
            print(f"subset {self.cfg.subset_name or 'extra.subset_ids'}: kept "
                  f"{len(self._clips)}/{before} clips")

        if self.cfg.filter_count_max:
            self._clips = self._clips[: self.cfg.filter_count_max]

        # Load scoreboard DB if it exists: {video_name: (sorted_frame_idxs, {fi: entry})}
        self._scoreboards: dict[str, tuple[list, dict]] = _load_scoreboards(
            self._path_preprocess / "scoreboards.db"
        )

        if (self.cfg.filter_score_zero or self.cfg.extra.get("filter_score_zero")) and self._scoreboards:
            before = len(self._clips)
            self._clips = [
                c for c in self._clips
                if not _clip_has_zero_score(c, self._scoreboards)
            ]
            print(
                f"filter_score_zero: dropped {before - len(self._clips)} clips "
                f"({len(self._clips)} remaining)"
            )

    def _ingest_split(
        self,
        split: str,
        names: list[str],
        frame_stride: int,
        scene_len: int,
        clip_stride: int,
        manifest: dict[str, int],
    ) -> None:
        for name in names:
            anno_file  = self._path_annotations / split / f"{name}.json"
            video_file = self._path_videos      / split / f"{name}.mp4"

            if not anno_file.exists() or not video_file.exists():
                continue

            with anno_file.open() as fh:
                raw = json.load(fh)
            self._annotations[name] = {int(k): v for k, v in raw.items()}

            ball_file = self._path_annotations / split / f"{name}_ball.json"
            if ball_file.exists():
                with ball_file.open() as fh:
                    raw_ball = json.load(fh)
                self._balls[name] = {
                    int(k): (None if v.get("x", -1) == -1 else (int(v["x"]), int(v["y"])))
                    for k, v in raw_ball.items()
                }

            n_frames = manifest.get(name) or _video_frame_count(video_file)
            window   = scene_len * frame_stride

            for start in range(0, n_frames - window + 1, clip_stride):
                indices = [start + i * frame_stride for i in range(scene_len)]
                self._clips.append(
                    {
                        "name":    name,
                        "split":   split,
                        "path":    video_file,
                        "indices": indices,
                    }
                )

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Scene:
        clip   = self._clips[idx]
        name   = clip["name"]
        anno   = self._annotations.get(name, {})
        rgbs   = _extract_frames(clip["path"], clip["indices"])
        events = [anno.get(fi) for fi in clip["indices"]]

        scoreboards = None
        if self._scoreboards:
            scoreboards = [
                _nearest_scoreboard(self._scoreboards, name, fi)
                for fi in clip["indices"]
            ]

        ball_anno = self._balls.get(name, {})
        balls = [ball_anno.get(fi) for fi in clip["indices"]]

        return Scene(
            scene_id    = f"{name}_{clip['indices'][0]:07d}",
            rgbs        = rgbs,
            events      = events,
            scoreboards = scoreboards,
            balls       = balls,
        )

    # ── collation ──────────────────────────────────────────────────────────────

    def collate_fn(self, samples: list[Scene]) -> SceneBatch:
        return collate_scenes(samples)

    # ── CLI hooks ──────────────────────────────────────────────────────────────

    @classmethod
    def preprocess(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cpu",
        video: Optional[str] = None,
        annotate: bool = False,
        override: bool = False,
        debug: bool = False,
        remove: bool = False,
    ) -> None:
        """Annotate per-video score bboxes interactively, then extract scores via VLM.

        Two-phase workflow:
          --annotate  Draw the scoreboard / left-score / right-score bboxes for
                      each video once (saved to video_bboxes.json).
          (default)   Use the saved bboxes and a VLM to read scores on every
                      sampled frame and store results in scoreboards.db.

        Args:
            cfg:      dataset configuration (paths, split, frame_stride, …).
            db:       SQLite output path; defaults to {path_preprocess}/scoreboards.db.
            model_id: HuggingFace model ID for VLM score reading.
            device:   torch device for VLM inference, e.g. "cuda:0".
            video:    restrict to a single video name, e.g. "game_1".
            annotate: if True, run interactive bbox annotation instead of VLM.
            override: re-annotate / re-process already-handled items.
            debug:    show crops and raw VLM output during processing.
        """
        from o3b.dataset.opentt.preprocess import (
            run_preprocess_from_dataset, BBOXES_FILENAME,
        )

        path_preprocess = Path(cfg.path_preprocess) if cfg.path_preprocess else Path(cfg.path_raw)
        db_path         = Path(db) if db else path_preprocess / "scoreboards.db"
        bboxes_path     = path_preprocess / BBOXES_FILENAME

        dataset = cls(cfg)

        if video:
            dataset._clips = [c for c in dataset._clips if c["name"] == video]

        if not dataset._clips:
            print(
                "No clips found. Run 'o3b dataset fetch -d opentt' first.",
                file=sys.stderr,
            )
            return

        frame_stride = int(cfg.extra.get("frame_stride", 1))
        n_videos     = len({c["name"] for c in dataset._clips})
        phase        = "annotating" if annotate else "VLM preprocessing"
        print(
            f"Scoreboard {phase}: {n_videos} video(s), "
            f"stride={frame_stride}"
            + (f"  →  {db_path}" if not annotate else f"  →  {bboxes_path}")
        )
        run_preprocess_from_dataset(
            dataset, db_path, bboxes_path,
            model_id=model_id, device=device,
            annotate=annotate, override=override, debug=debug,
            remove=remove,
        )

    @classmethod
    def fetch(cls, cfg: DatasetConfig, *, url: Optional[str] = None) -> None:
        """Download annotation JSONs from GitHub and MP4 videos from the CDN."""
        path_raw = cfg.path_raw or cfg.root / "opentt"

        print("=== OpenTT: fetching annotations ===")
        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            anno_dir = path_raw / "annotations" / split
            anno_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = anno_dir / f"{name}.json"
                if dest.exists():
                    print(f"  skip  annotations/{split}/{name}.json  (already exists)")
                    continue
                src = f"{_ANNO_RAW}/{split}/{name}.json"
                print(f"  fetch {src}")
                urllib.request.urlretrieve(src, dest, _progress)
                print()

        print("\n=== OpenTT: fetching ball annotations ===")
        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            anno_dir = path_raw / "annotations" / split
            anno_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = anno_dir / f"{name}_ball.json"
                if dest.exists():
                    print(f"  skip  annotations/{split}/{name}_ball.json  (already exists)")
                    continue
                src = f"{_VIDEO_BASE}/{name}.zip"
                tmp = anno_dir / f"{name}_ball.zip.tmp"
                print(f"  fetch {src}  (extracting ball_markup.json)")
                try:
                    urllib.request.urlretrieve(src, tmp, _progress)
                    print()
                    with zipfile.ZipFile(tmp) as zf:
                        with zf.open("ball_markup.json") as bf:
                            ball_data = json.load(bf)
                    with dest.open("w") as fh:
                        json.dump(ball_data, fh)
                    print(f"  saved → {dest}")
                finally:
                    if tmp.exists():
                        tmp.unlink()

        print("\n=== OpenTT: fetching videos ===")
        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            vid_dir = path_raw / "videos" / split
            vid_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                dest = vid_dir / f"{name}.mp4"
                if dest.exists():
                    print(f"  skip  videos/{split}/{name}.mp4  (already exists)")
                    continue
                src = f"{_VIDEO_BASE}/{name}.mp4"
                print(f"  fetch {src}")
                urllib.request.urlretrieve(src, dest, _progress)
                print()

        print("\nDone. Run 'o3b dataset index -d opentt' next.")

    @classmethod
    def index(cls, cfg: DatasetConfig, *, db: Optional[Path] = None) -> None:
        """Scan all available videos, record frame counts, write manifest.json.

        The manifest lets _setup() skip re-opening every mp4 file on subsequent
        dataset instantiations.
        """
        path_raw        = cfg.path_raw        or cfg.root / "opentt"
        path_preprocess = cfg.path_preprocess or path_raw
        manifest_path   = db or Path(path_preprocess) / "manifest.json"

        Path(path_preprocess).mkdir(parents=True, exist_ok=True)

        manifest: dict[str, int] = {}
        total_clips = 0

        frame_stride: int = int(cfg.extra.get("frame_stride", 1))
        scene_len:    int = cfg.scene_length
        clip_stride:  int = int(cfg.extra.get("clip_stride", scene_len * frame_stride))

        print(f"{'Video':<20}  {'Split':<6}  {'Frames':>8}  {'Clips':>7}")
        print("-" * 50)

        for split, names in [("train", _TRAIN_NAMES), ("test", _TEST_NAMES)]:
            vid_dir  = Path(path_raw) / "videos"  / split
            anno_dir = Path(path_raw) / "annotations" / split

            for name in names:
                video_file = vid_dir  / f"{name}.mp4"
                anno_file  = anno_dir / f"{name}.json"

                if not video_file.exists():
                    print(f"  {'(missing)':20}  {split:<6}  {'—':>8}  {'—':>7}  {name}")
                    continue

                n_frames = _video_frame_count(video_file)
                manifest[name] = n_frames

                window     = scene_len * frame_stride
                n_clips    = max(0, (n_frames - window) // clip_stride + 1)
                total_clips += n_clips

                anno_ok = "✓" if anno_file.exists() else "✗ no annotation"
                print(f"  {name:<20}  {split:<6}  {n_frames:>8,}  {n_clips:>7,}  {anno_ok}")

        print("-" * 50)
        print(f"  {'TOTAL':<20}  {'':6}  {'':8}  {total_clips:>7,}")

        with manifest_path.open("w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nManifest written → {manifest_path}")

    @classmethod
    def visualize(
        cls,
        cfg: DatasetConfig,
        *,
        db: Optional[Path] = None,
        limit: int = 4,
        object_id: Optional[str] = None,
        frame_stride: Optional[int] = None,
        frames_per_scene: Optional[int] = None,
        render: bool = False,
        render_frames: int = 0,
        renderer: str = "pyrender",
        debug: bool = False,
    ) -> None:
        """Load and play up to *limit* clips, one at a time.

        Each clip's frames are decoded upfront, then shown in an OpenCV
        player.  *object_id* restricts clips to a specific video name
        (e.g. --object-id game_1).

        When *frames_per_scene* is set, only that many evenly-sampled frames
        are loaded and shown as a static grid — no interactive player.
        """
        dataset = cls(cfg)

        if not dataset._clips:
            print(
                "No clips found. Run 'o3b dataset fetch -d opentt' first, then make sure\n"
                f"  path_raw → {dataset._path_raw}\n"
                "contains annotations/ and videos/ sub-directories.",
                file=sys.stderr,
            )
            return

        clips = dataset._clips
        if object_id:
            clips = [c for c in clips if c["name"] == object_id]
            if not clips:
                print(
                    f"No clips found for video '{object_id}'. "
                    f"Available: {sorted({c['name'] for c in dataset._clips})}",
                    file=sys.stderr,
                )
                return

        clips = clips[:limit]
        print(
            f"Visualising {len(clips)} clip(s)"
            + (f" (filtered to '{object_id}')" if object_id else "")
            + f"  [{dataset._path_raw}]"
        )

        ds_frame_stride = int(cfg.extra.get("frame_stride", 1))
        for clip_idx, clip in enumerate(clips):
            name    = clip["name"]
            indices = clip["indices"]
            anno    = dataset._annotations.get(name, {})

            # Sub-sample indices when --frames-per-scene is set
            if frames_per_scene is not None and frames_per_scene < len(indices):
                step     = max(1, (len(indices) - 1) // (frames_per_scene - 1)) if frames_per_scene > 1 else len(indices)
                sampled  = [indices[min(i * step, len(indices) - 1)] for i in range(frames_per_scene)]
                # remove duplicates while preserving order
                seen: set = set()
                sampled = [fi for fi in sampled if not (fi in seen or seen.add(fi))]  # type: ignore[func-returns-value]
            else:
                sampled = indices

            print(f"\n[{clip_idx + 1}/{len(clips)}]  {name}  frames {indices[0]}–{indices[-1]}")
            print(f"  Decoding {len(sampled)} frame(s)…", end=" ", flush=True)
            rgbs = _extract_frames(clip["path"], sampled)
            if rgbs is None:
                print("failed — skipping.")
                continue
            print("done")

            events      = [anno.get(fi) for fi in sampled]
            scoreboards = [
                _nearest_scoreboard(dataset._scoreboards, name, fi)
                for fi in sampled
            ] if dataset._scoreboards else [None] * len(sampled)

            ball_anno = dataset._balls.get(name, {})
            balls = [ball_anno.get(fi) for fi in sampled]

            if frames_per_scene is not None:
                _viz_grid(
                    scene_id      = f"{name}_{indices[0]:07d}",
                    rgbs          = rgbs,
                    events        = events,
                    scoreboards   = scoreboards,
                    balls         = balls,
                    frame_indices = sampled,
                    video_fps     = 120.0,
                )
            else:
                _viz_scene(
                    scene_id    = f"{name}_{indices[0]:07d}",
                    rgbs        = rgbs,
                    events      = events,
                    scoreboards = scoreboards,
                    balls       = balls,
                    fps         = max(1.0, 120.0 / ds_frame_stride),
                )


# ── viewers ───────────────────────────────────────────────────────────────────

def _viz_grid(
    scene_id: str,
    rgbs: "torch.Tensor",
    events: list,
    scoreboards: list,
    balls: "list | None" = None,
    frame_indices: "list[int] | None" = None,
    video_fps: float = 1.0,
) -> None:
    """Show a static grid of frames (one tile per frame).

    Tile size is derived from the screen so the window fills 2/3 of the
    display without upscaling (pixels are never invented).

    Controls:  any key → next clip   Q / Esc → quit all
    """
    import math
    import cv2
    import numpy as np

    T = rgbs.shape[0]
    if T == 0:
        return

    cols = math.ceil(math.sqrt(T))
    rows = math.ceil(T / cols)

    # Determine available display area (2/3 of screen, fall back to 1280×720)
    disp_w, disp_h = 1280, 720
    try:
        import tkinter as _tk
        _root = _tk.Tk()
        disp_w = int(_root.winfo_screenwidth()  * 2 / 3)
        disp_h = int(_root.winfo_screenheight() * 2 / 3)
        _root.destroy()
    except Exception:
        pass

    # Tile dimensions: fit the grid into the display area, never upscale
    orig_h, orig_w = rgbs.shape[1], rgbs.shape[2]
    tile_w = min(orig_w, disp_w // cols)
    tile_h = min(orig_h, disp_h // rows)
    # Preserve aspect ratio: scale by the more constraining dimension
    scale  = min(tile_w / orig_w, tile_h / orig_h)
    tile_w = int(orig_w * scale)
    tile_h = int(orig_h * scale)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    tiles = []
    for t in range(T):
        frame_rgb = (rgbs[t].clamp(0, 1).numpy() * 255).astype(np.uint8)
        tile      = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if (tile_w, tile_h) != (orig_w, orig_h):
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        tw, th = tile_w, tile_h

        fscale    = max(tw, th) / 700.0
        thickness = max(1, round(fscale * 2))
        margin    = max(4, int(tw * 0.015))
        y_pos     = max(16, int(th * 0.07))

        def _put(text: str, x: int, y: int, color: tuple, _tile: "np.ndarray" = tile) -> None:
            cv2.putText(_tile, text, (x, y), font, fscale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(_tile, text, (x, y), font, fscale, color,     thickness,     cv2.LINE_AA)

        sb          = scoreboards[t] if scoreboards[t] is not None else {}
        left_score  = sb.get("score_left_point")  if sb else None
        right_score = sb.get("score_right_point") if sb else None
        l_str = str(left_score)  if left_score  is not None else "?"
        r_str = str(right_score) if right_score is not None else "?"

        _put(f"L:{l_str}", margin, y_pos, (80, 220, 80))
        r_label_w = cv2.getTextSize(f"R:{r_str}", font, fscale, thickness)[0][0]
        _put(f"R:{r_str}", tw - r_label_w - margin, y_pos, (80, 80, 220))

        ev = events[t]
        if ev:
            ev_tw = cv2.getTextSize(ev, font, fscale * 0.65, thickness)[0][0]
            cv2.putText(tile, ev, ((tw - ev_tw) // 2, th - margin),
                        font, fscale * 0.65, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(tile, ev, ((tw - ev_tw) // 2, th - margin),
                        font, fscale * 0.65, (220, 220, 80), thickness, cv2.LINE_AA)

        if frame_indices is not None:
            secs   = frame_indices[t] / video_fps
            fi_str = f"{int(secs // 60)}:{secs % 60:05.2f}"
            _put(fi_str, margin, th - margin, (200, 200, 200))

        ball = balls[t] if (balls and t < len(balls)) else None
        if ball is not None:
            bx = int(ball[0] * tile_w / orig_w)
            by = int(ball[1] * tile_h / orig_h)
            radius = max(4, int(min(tile_w, tile_h) * 0.015))
            cv2.circle(tile, (bx, by), radius + 1, (0, 0, 0),   -1, cv2.LINE_AA)
            cv2.circle(tile, (bx, by), radius,     (0, 255, 255), -1, cv2.LINE_AA)

        tiles.append(tile)

    # Pad to fill the grid
    blank = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank)

    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid      = np.vstack(grid_rows)

    win = scene_id
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setWindowTitle(win, f"{scene_id}  ({T} frames)  —  any key: next   Q/Esc: quit")
    cv2.imshow(win, grid)
    print(f"  {scene_id}  ({T} frames)  —  any key: next clip   Q/Esc: quit")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(win)
    if key in (27, ord('q'), ord('Q')):
        sys.exit(0)


def _viz_scene(
    scene_id: str,
    rgbs: "torch.Tensor",
    events: list,
    scoreboards: list,
    balls: "list | None" = None,
    fps: float = 30.0,
) -> None:
    """Play a single pre-loaded clip in an OpenCV window.

    Controls:
      Space       play / pause
      ← / →  or  A / D    step one frame
      Q / Esc              close and move to next clip
    """
    import cv2
    import numpy as np

    T = rgbs.shape[0]
    if T == 0:
        return

    win = scene_id
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    current = [0]
    playing = [False]

    def on_frame(pos: int) -> None:
        current[0] = pos

    cv2.createTrackbar("Frame", win, 0, max(T - 1, 1), on_frame)
    print(
        f"  {scene_id}  ({T} frames @ {fps:.0f} fps)  —  "
        "Space: play/pause   ←→/AD: step   Q/Esc: next clip"
    )

    font      = None   # resolved on first frame
    scale     = None
    thickness = None
    margin    = None
    y_pos     = None
    line_h    = None

    while True:
        t = current[0]

        # Convert (H, W, 3) float32 → uint8 BGR
        frame_rgb = (rgbs[t].clamp(0, 1).numpy() * 255).astype(np.uint8)
        display   = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        H, W      = display.shape[:2]

        # Lazily compute text metrics on first frame
        if font is None:
            font      = cv2.FONT_HERSHEY_SIMPLEX
            scale     = max(W, H) / 700.0
            thickness = max(1, round(scale * 2))
            margin    = max(8, int(W * 0.015))
            y_pos     = max(20, int(H * 0.06))
            line_h    = int(y_pos * 1.4)

        def _put(text: str, x: int, y: int, color: tuple) -> None:
            cv2.putText(display, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(display, text, (x, y), font, scale, color,     thickness,     cv2.LINE_AA)

        # Scoreboard overlay
        sb          = scoreboards[t] if scoreboards[t] is not None else {}
        left_score  = sb.get("score_left_point")  if sb else None
        right_score = sb.get("score_right_point") if sb else None
        l_str = str(left_score)  if left_score  is not None else "?"
        r_str = str(right_score) if right_score is not None else "?"

        _put(f"L: {l_str}", margin, y_pos, (80, 220, 80))
        r_tw = cv2.getTextSize(f"R: {r_str}", font, scale, thickness)[0][0]
        _put(f"R: {r_str}", W - r_tw - margin, y_pos, (80, 80, 220))

        # Event label centred on second row
        ev  = events[t]
        ev_str = ev if ev is not None else ""
        if ev_str:
            ev_tw = cv2.getTextSize(ev_str, font, scale * 0.75, thickness)[0][0]
            cv2.putText(display, ev_str, ((W - ev_tw) // 2, y_pos + line_h),
                        font, scale * 0.75, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(display, ev_str, ((W - ev_tw) // 2, y_pos + line_h),
                        font, scale * 0.75, (220, 220, 80), thickness, cv2.LINE_AA)

        ball = balls[t] if (balls and t < len(balls)) else None
        if ball is not None:
            radius = max(6, int(min(W, H) * 0.012))
            cv2.circle(display, (ball[0], ball[1]), radius + 2, (0, 0, 0),    -1, cv2.LINE_AA)
            cv2.circle(display, (ball[0], ball[1]), radius,     (0, 255, 255), -1, cv2.LINE_AA)

        cv2.setWindowTitle(win, f"{scene_id}  |  frame {t + 1}/{T}  |  L:{l_str}  R:{r_str}")
        cv2.imshow(win, display)
        cv2.setTrackbarPos("Frame", win, t)

        delay = max(1, int(1000 / fps)) if playing[0] else 30
        key   = cv2.waitKey(delay) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            playing[0] = not playing[0]
        elif key in (83, 3, ord('d'), ord('D')):    # → or D
            playing[0] = False
            current[0] = min(t + 1, T - 1)
        elif key in (81, 2, ord('a'), ord('A')):    # ← or A
            playing[0] = False
            current[0] = max(t - 1, 0)
        elif playing[0]:
            nxt = t + 1
            if nxt >= T:
                playing[0] = False
            else:
                current[0] = nxt

    cv2.destroyWindow(win)


# ── helpers ───────────────────────────────────────────────────────────────────

def _video_frame_count(path: Path) -> int:
    """Return total frame count for an mp4 via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def _extract_frames(path: Path, indices: list[int]) -> Optional[torch.Tensor]:
    """Seek to each index in an mp4 and return (T, H, W, 3) float32 in [0, 1]."""
    import cv2
    cap    = cv2.VideoCapture(str(path))
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).float() / 255.0)
    cap.release()
    return torch.stack(frames) if frames else None


def _load_scoreboards(db_path: Path) -> dict:
    """Load a scoreboards SQLite DB into memory for O(log n) per-frame lookup.

    Returns {video_name: (sorted_frame_idxs, {frame_idx: entry_dict})}
    where entry_dict may contain:
      "bbox"             – whole scoreboard region
      "score_left"       – left point score (backward-compat)
      "score_right"      – right point score (backward-compat)
      "bbox_left_point"  – bbox of left point score (big number)
      "score_left_point"
      "bbox_left_game"   – bbox of left game score (small number)
      "score_left_game"
      "bbox_right_point" – bbox of right point score (big number)
      "score_right_point"
      "bbox_right_game"  – bbox of right game score (small number)
      "score_right_game"
    Returns {} if *db_path* does not exist.
    """
    if not db_path.exists():
        return {}

    import sqlite3

    con = sqlite3.connect(db_path, timeout=30)
    try:
        existing_cols = {row[1] for row in con.execute("PRAGMA table_info(scoreboards)")}

        base_cols = [
            "video_name", "frame_idx",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "score_left", "score_right",
        ]
        extra_cols = [
            "bbox_left_point_x1",  "bbox_left_point_y1",
            "bbox_left_point_x2",  "bbox_left_point_y2",  "score_left_point",
            "bbox_left_game_x1",   "bbox_left_game_y1",
            "bbox_left_game_x2",   "bbox_left_game_y2",   "score_left_game",
            "bbox_right_point_x1", "bbox_right_point_y1",
            "bbox_right_point_x2", "bbox_right_point_y2", "score_right_point",
            "bbox_right_game_x1",  "bbox_right_game_y1",
            "bbox_right_game_x2",  "bbox_right_game_y2",  "score_right_game",
        ]
        all_cols = base_cols + [c for c in extra_cols if c in existing_cols]

        rows = con.execute(
            f"SELECT {', '.join(all_cols)} FROM scoreboards"
        ).fetchall()
    finally:
        con.close()

    raw: dict[str, dict[int, dict]] = {}
    for row in rows:
        r = dict(zip(all_cols, row))
        video_name = r["video_name"]
        fi = int(r["frame_idx"])

        entry: dict = {}
        if r.get("bbox_x1") is not None:
            entry["bbox"] = [r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"]]
        if r.get("score_left") is not None:
            entry["score_left"] = int(r["score_left"])
        if r.get("score_right") is not None:
            entry["score_right"] = int(r["score_right"])

        for region in ("left_point", "left_game", "right_point", "right_game"):
            x1k = f"bbox_{region}_x1"
            if r.get(x1k) is not None:
                entry[f"bbox_{region}"] = [
                    r[f"bbox_{region}_x1"], r[f"bbox_{region}_y1"],
                    r[f"bbox_{region}_x2"], r[f"bbox_{region}_y2"],
                ]
            sc = r.get(f"score_{region}")
            if sc is not None:
                entry[f"score_{region}"] = int(sc)

        raw.setdefault(video_name, {})[fi] = entry

    result: dict[str, tuple] = {}
    for name, frame_dict in raw.items():
        result[name] = (sorted(frame_dict.keys()), frame_dict)

    return result


def _nearest_scoreboard(
    scoreboards: dict,
    name: str,
    fi: int,
    max_dist: int = 45,
) -> Optional[dict]:
    """Return the nearest stored scoreboard entry for frame *fi*, or None.

    Searches within *max_dist* frames of *fi* (default 45 ≈ 1.5 × stride=30).
    """
    entry = scoreboards.get(name)
    if entry is None:
        return None
    sorted_keys, frame_dict = entry
    if not sorted_keys:
        return None

    import bisect

    pos = bisect.bisect_left(sorted_keys, fi)
    candidates: list[int] = []
    if pos < len(sorted_keys):
        candidates.append(sorted_keys[pos])
    if pos > 0:
        candidates.append(sorted_keys[pos - 1])

    nearest = min(candidates, key=lambda k: abs(k - fi))
    if abs(nearest - fi) <= max_dist:
        return frame_dict[nearest]
    return None


def _clip_has_zero_score(clip: dict, scoreboards: dict) -> bool:
    """Return True if the clip's middle frame has a known scoreboard with both scores == 0."""
    indices = clip["indices"]
    mid_fi  = indices[len(indices) // 2]
    sb = _nearest_scoreboard(scoreboards, clip["name"], mid_fi)
    if sb is None:
        return False
    left  = sb.get("score_left_point")  if sb.get("score_left_point")  is not None else sb.get("score_left")
    right = sb.get("score_right_point") if sb.get("score_right_point") is not None else sb.get("score_right")
    return left == 0 and right == 0


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """urlretrieve reporthook — inline progress bar."""
    if total_size <= 0:
        sys.stdout.write(f"\r  {block_num * block_size / 1e6:.1f} MB")
    else:
        pct  = min(100.0, block_num * block_size * 100.0 / total_size)
        done = int(pct / 2)
        sys.stdout.write(f"\r  [{'=' * done}{' ' * (50 - done)}] {pct:5.1f}%")
    sys.stdout.flush()
