"""Shard-wise shuffling sampler — od3d's ``ShardAndSampleShuffledSampler``.

The index range is cut into consecutive blocks of ``shard_size`` items (od3d's
fixed 1048, which is not the on-disk Arrow shard size). Per epoch the block
order is permuted (``shuffle_shards``) and, independently, the order inside each
block (``shuffle_within_shards``); each rank then takes a contiguous quarter (or
1/world_size) of that list, or every world_size-th index with ``consecutive``.

Against a global permutation (DistributedSampler) this keeps a batch inside one
block. Sharded caches store items in category/sequence order, so a batch then
holds one or two categories and a few sequences — consecutive frames of them
when ``shuffle_within_shards`` is off, which is what od3d's Trinity configs
train with (``omni6dpose_bbox_trinity.yaml``: shards True, within False).
Ported so a run can reproduce that sampling rather than o3b's default.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class ShardShuffleSampler(Sampler):
    def __init__(self, n: int, rank: int = 0, world_size: int = 1, seed: int = 42,
                 shard_size: int = 1048, shuffle_shards: bool = True,
                 shuffle_within_shards: bool = True, consecutive: bool = False,
                 drop_last: bool = True):
        self.n = int(n)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed, self.shard_size = int(seed), int(shard_size)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shards = bool(shuffle_within_shards)
        self.consecutive, self.drop_last = bool(consecutive), bool(drop_last)
        self.epoch = 0
        self._starts = np.arange(0, self.n, self.shard_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(len(self._starts))
        if self.shuffle_shards:
            rng.shuffle(order)
        parts = []
        for s in order:
            start = self._starts[s]
            idx = np.arange(start, min(start + self.shard_size, self.n))
            if self.shuffle_within_shards:
                rng.shuffle(idx)
            parts.append(idx)
        perm = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        per_rank = (len(perm) // self.world_size if self.drop_last
                    else -(-len(perm) // self.world_size))
        if self.consecutive:
            mine = perm[self.rank::self.world_size]
        else:
            mine = perm[self.rank * per_rank:(self.rank + 1) * per_rank]
        return mine[:per_rank]

    def __iter__(self):
        return iter(int(i) for i in self._indices())

    def __len__(self) -> int:
        return (self.n // self.world_size if self.drop_last
                else -(-self.n // self.world_size))
