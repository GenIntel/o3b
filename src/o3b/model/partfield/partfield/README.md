# Vendored PartField (feature-extraction subset)

Source: https://github.com/nv-tlabs/PartField
Commit: `373025dbd283bb44cc4a6dc78c99994dbc91de32`
License: NVIDIA non-commercial (see `LICENSE`).

Only the modules needed to extract per-vertex features are vendored
(PVCNN point encoder → triplane transformer → triplane feature sampling):

- `triplane.py`, `model_utils.py` — upstream `partfield/model/`
- `PVCNN/` — upstream `partfield/model/PVCNN/` minus training-only files
  (`pv_module/{pointnet,ball_query,loss,frustum}.py` are unused at inference
  and were dropped)

Modifications:

- `PVCNN/dnnlib_util.py` replaced by a no-op stub (the real file needs
  boto3/loguru/psutil; only `ScopedTorchProfiler`/`printarr` are imported,
  and only from commented-out code).

The Lightning trainer wrapper (`model_trainer_pvcnn_only_demo.py`) is NOT
vendored; `o3b/model/partfield/model.py` replicates its `predict_step`
without the PyTorch Lightning dependency. External deps: `torch-scatter`,
`einops`.
