# Vendored MeshFM (feature-extraction subset)

Source: https://github.com/threedle/MeshFM
Commit: `e6577b15e4913895833801fd1154399cec3955d2`
Paper: Zhou, Liu, Lang, Hanocka — *MeshFM: 2D Features Are All You Need for
3D Shape Understanding*, ECCV 2026.
License: **none published** — the upstream repository ships no LICENSE file,
so it is all-rights-reserved by default. Vendored for internal research use;
do not redistribute without clearing that with the authors.

Only the modules needed to extract per-vertex features are vendored
(PVCNN point encoder → triplane transformer → tanh → triplane sampling):

- `model/` — upstream `meshfm/model/` unchanged (`SimpleTriplaneModel`,
  `TriplaneTransformer`, `pvcnn/`)
- `_scatter.py` — upstream's pure-torch `scatter_mean`, which is why MeshFM,
  unlike PartField, needs no `torch-scatter` build
- `sampling.py` — upstream's area-weighted surface sampler

Not vendored: `inference.py`, `mesh_io.py`, `viz.py`, `config.py`,
`scripts/`. `o3b/model/meshfm/model.py` replicates
`MeshFM.extract_mesh_features(mode='vertex')` — unit-sphere normalisation,
100k surface samples, query clamped to [-1, 1] — and inlines the one config
upstream reads from `configs/meshfm_1536d.yaml`.

Checkpoint: `meshfm.pth` (502 MB, a plain state dict, 125.5 M params) on Google
Drive, id `1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc`; fetched with `gdown` on first
use. External deps: `einops`, `gdown`.
