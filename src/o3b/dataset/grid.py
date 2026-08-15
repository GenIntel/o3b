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

# Budget for ~/.o3b/cache/axes.  A page is ~15 MB of uncompressed uint8 and the
# prefetcher fills pages that may never be opened, so the cache is pruned
# (least recently used first) once it passes this.
CACHE_MAX_BYTES = 4 * 1024 ** 3
