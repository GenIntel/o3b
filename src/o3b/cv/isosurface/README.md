# Isosurface remeshing: DMTet and FlexiCubes

Used by the `dmtet<N>` / `fc<N>` mesh types (`o3b/data/datatypes/mesh.py`),
the counterparts of `mc<N>`. `remesh.py` is o3b's; the rest is vendored.

Source: https://github.com/NVlabs/nvdiffrec, commit `abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb`
License: NVIDIA Source Code License for nvdiffrec (`LICENSE_nvdiffrec.txt`) —
non-commercial / research use only.

- `flexicubes.py`, `tables.py` — `geometry/flexicubes.py` and `geometry/tables.py`,
  unchanged except the `tables` import made relative.
- `dmtet.py` — the marching-tets extraction and `sdf_reg_loss` of
  `geometry/dmtet.py` (itself adapted from kaolin); the device comes from the
  inputs instead of a hard-coded `"cuda"`, and `map_uv` / `DMTetGeometry` /
  the renderer coupling are dropped.

## From a source mesh to `dmtet16` / `fc16`

Same box, same samples and same finish as `mc16`, so the three differ in the
extraction alone:

1. **Grid**: the source bbox padded by 5 %, 16 SDF samples per axis — for
   FlexiCubes 15^3 cubes, for DMTet the same cubes each split into 6 Kuhn
   tetrahedra along the main diagonal (conforming; nvdiffrec ships only
   quartet grids at 32/64/128, all positively oriented, as these are).
2. **SDF**: `igl.signed_distance` at the grid points, with `mc16`'s surface
   snapping (points within half a cell of the surface move onto it at SDF
   -1e-5, keeping thin features from vanishing).
3. **Fit** (`_o<steps>`, default 300; `_o0` skips it): Adam (lr 5e-3) on the
   SDF values, a per-vertex deformation bounded to a quarter edge
   (`tanh`), and — FlexiCubes — the 21 per-cube weights, against the
   two-sided chamfer distance between 10k points sampled differentiably on
   the extracted surface and 10k on the source, plus nvdiffrec's SDF
   regulariser (0.2 → 0.01 over the first quarter) and FlexiCubes' L_dev.
   nvdiffrec fits the same parameters against a rendering loss because it
   only has images; the ground-truth surface makes a 3-D loss possible here.
4. **Extract** (FlexiCubes with `training=False`), wind like
   `igl.marching_cubes`, then `mc16`'s finish: clamp and stretch onto the
   source box, texture atlas from the source colours.

The fitted geometry is cached bare under `mesh/<dmtet16>/` before features
are added, so every feature type of an object shares one fit.
