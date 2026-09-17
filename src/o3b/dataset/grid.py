"""Page shape shared by the two grid editors (``axes-tform-obj-type``,
``select-subset``).

Objects run across, their own views down: ``GRID_COLS`` objects side by side
form a block ``GRID_VIEWS`` rows tall, and ``GRID_OBJECTS`` fills two such
blocks — a 6 x 8 grid of 16 objects with three views each, which is about as
much of a category as one screen can show and still be judged.

Its own module, holding nothing but the numbers, so ``o3b/cli.py`` can name
these as argparse defaults without importing the editors (and through them
torch) on every ``o3b`` invocation.
"""
from __future__ import annotations

GRID_COLS = 8
GRID_VIEWS = 3
GRID_OBJECTS = GRID_COLS * 2

# How far ahead the editors warm the crop cache in the background: this many
# pages of the current category, then this many coming categories.  Three of
# each is about a minute of decoding, which is far less than a minute of
# judging 48 thumbnails.
PREFETCH = 3

# Budget for ~/.o3b/cache/axes.  A page is ~0.5 MB of JPEG and the prefetcher
# fills pages that may never be opened, so the cache is pruned (least recently
# used first) once it passes this.  Room for every category's first page many
# times over — at the raw uint8 the pages used to be stored as, ~15 MB each,
# this held 300 of them against 965 categories and evicted what it had warmed.
CACHE_MAX_BYTES = 4 * 1024 ** 3

# ── contact sheet (o3b dataset viz-cat, see o3b/dataset/viz_category.py) ──────
# A sheet is one category: SHEET_OBJECTS objects down, SHEET_FRAMES frames of
# each across, in timestamp order. Here rather than in viz_category.py for the
# same reason the grid constants are: o3b/cli.py names them as argparse
# defaults, and must be able to import them without pulling in torch.
#
# RECONSTRUCTED after an accidental `git checkout -- .` discarded the working
# tree. SHEET_OBJECTS / SHEET_FRAMES / SHEET_CELL are recovered exactly, from
# run_category_sheet's own defaults (n_objects=8, n_frames=8) and its _CELL=320.
# SHEET_GAP and SHEET_MARGIN are NOT recovered values — only their roles are
# known (gap = pixels of white ground between cells; margin = the crop, "tight"
# against the editors' 0.45). Check these two against what the sheets should
# look like.
SHEET_OBJECTS = 8
SHEET_FRAMES = 8
SHEET_CELL = 320        # what _crop_frame renders; --cell only scales it down
SHEET_GAP = 8           # reconstructed
SHEET_MARGIN = 0.05     # reconstructed; the editors use 0.45
