"""sharded_rows: a cache read back through a row list (order kept, repeats allowed)."""
import numpy as np
import pytest

from o3b.dataset.dataset import ConfigurableDataset, DatasetConfig


class _Rows(ConfigurableDataset):
    def _setup_sharded(self):
        from datasets import Dataset
        self._sharded = Dataset.from_dict({"i": list(range(10))})


def _cfg(tmp_path, rows):
    return DatasetConfig.from_dict({
        "class_name": "x", "path_preprocess": str(tmp_path),
        "sharded_name": "cache", "sharded_rows": rows,
    })


def test_relative_rows_file(tmp_path):
    d = tmp_path / "sharded" / "cache.rows"
    d.mkdir(parents=True)
    np.save(d / "pick.npy", np.array([3, 3, 7, 0]))
    ds = _Rows(_cfg(tmp_path, "pick"))
    assert len(ds) == 4
    assert ds._sharded["i"] == [3, 3, 7, 0]


def test_absolute_rows_file_and_bounds(tmp_path):
    np.save(tmp_path / "bad.npy", np.array([10]))
    with pytest.raises(ValueError):
        _Rows(_cfg(tmp_path, str(tmp_path / "bad.npy")))


def test_unset_reads_everything(tmp_path):
    assert len(_Rows(_cfg(tmp_path, None))) == 10
