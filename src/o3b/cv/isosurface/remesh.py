"""Remesh a source mesh through DMTet or FlexiCubes, fitted to the source surface.

The counterpart of ``convert_mesh_to_mc`` for the differentiable isosurfaces of
nvdiffrec (see README.md). Same grid box (source bbox + 5 %), same number of
SDF samples (``n`` per axis), same initial SDF (igl signed distance, with
mc's surface snapping), so the three differ in the extraction alone:

- ``dmtet``: marching tetrahedra on a Kuhn tet grid (each of the (n-1)^3
  cubes split into 6 tets along its main diagonal -- conforming, needs no
  quartet binary, which is how nvdiffrec builds its 32/64/128 grids).
- ``fc``: FlexiCubes, dual marching cubes with per-cube weights, on the
  (n-1)^3 cube grid.

Extracting once from the exact SDF would only re-triangulate the same samples,
so by default the extraction is *fitted*: the SDF values, a bounded per-vertex
grid deformation and (FlexiCubes) the per-cube weights are optimised with Adam
against the two-sided chamfer distance to the source surface, plus
nvdiffrec's SDF regulariser (and FlexiCubes' L_dev). nvdiffrec fits the same
parameters against a multi-view rendering loss because it has only images;
here the ground-truth surface is at hand, so the loss is taken in 3-D. That is
what lets the surface leave the grid for sharp edges and sub-voxel parts,
which marching cubes on the same samples cannot do.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# the 6 Kuhn tets of a cube, as paths 000 -> a -> a+b -> 111 over the axis
# permutations; corner index = x + 2y + 4z
_KUHN_TETS = [
    [0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
    [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7],
]


def kuhn_tet_grid(n: int):
    """(n^3, 3) vertices in [-0.5, 0.5]^3 and (6 (n-1)^3, 4) positively oriented tets."""
    u = np.linspace(-0.5, 0.5, n)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)

    i, j, k = np.meshgrid(*(np.arange(n - 1),) * 3, indexing="ij")
    base = (i * n * n + j * n + k).reshape(-1)
    corner = np.array([dx * n * n + dy * n + dz
                       for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)])  # x + 2y + 4z
    tets = (base[:, None, None] + corner[np.array(_KUHN_TETS)][None]).reshape(-1, 4)

    # orient like nvdiffrec's quartet grids: (b-a) x (c-a) . (d-a) > 0
    a, b, c, d = (verts[tets[:, m]] for m in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return verts, tets


def _unique_edges(cells: torch.Tensor, pairs) -> torch.Tensor:
    e = cells[:, torch.tensor(pairs, device=cells.device)].reshape(-1, 2)
    return torch.unique(torch.sort(e, dim=1)[0], dim=0)


def _sample_surface(verts, faces, n, generator=None):
    """Area-weighted surface samples, differentiable in ``verts``."""
    tri = verts[faces]
    areas = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    idx = torch.multinomial(areas.detach() + 1e-12, n, replacement=True, generator=generator)
    uv = torch.rand(n, 2, device=verts.device, generator=generator)
    su = uv[:, :1].sqrt()
    w1, w2 = su * (1.0 - uv[:, 1:]), su * uv[:, 1:]
    t = tri[idx]
    return t[:, 0] + w1 * (t[:, 1] - t[:, 0]) + w2 * (t[:, 2] - t[:, 0])


def _chamfer(a, b, chunk=4096):
    """Two-sided mean squared nearest-neighbour distance.

    The nearest neighbours are found without autograd (|p|^2 + |q|^2 - 2 p.q
    by matmul) and only the matched pairs are differentiated -- the same
    gradient as differentiating through the min, minus torch.cdist's backward,
    which was ~60 % of a fit step.
    """
    def one_way(p, q):
        with torch.no_grad():
            qq = q.pow(2).sum(1)
            idx = torch.cat([(qq[None] - 2 * p[i:i + chunk] @ q.T).argmin(dim=1)
                             for i in range(0, p.shape[0], chunk)])
        return (p - q[idx]).pow(2).sum(1).mean()
    return one_way(a, b) + one_way(b, a)


def chamfer_to(verts, faces, V, F, n=20000, seed=0) -> float:
    """Two-sided chamfer between two meshes (numpy), for comparing remeshes."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(seed)
    p = _sample_surface(torch.as_tensor(verts, dtype=torch.float32, device=dev),
                        torch.as_tensor(faces, dtype=torch.long, device=dev), n, g)
    q = _sample_surface(torch.as_tensor(V, dtype=torch.float32, device=dev),
                        torch.as_tensor(F, dtype=torch.long, device=dev), n, g)
    return float(_chamfer(p, q))


def remesh_isosurface(
    V: np.ndarray,
    F: np.ndarray,
    method: str,
    n: int = 16,
    fit_steps: int = 300,
    lr: float = 5e-3,
    n_samples: int = 10000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts, faces) of the ``dmtet`` / ``fc`` remesh of (V, F).

    Faces are wound like ``igl.marching_cubes`` on an inside-negative SDF, so
    the caller can finish exactly as ``convert_mesh_to_mc`` does.
    """
    import igl

    from o3b.cv.isosurface.dmtet import DMTet, sdf_reg_loss
    from o3b.cv.isosurface.flexicubes import FlexiCubes

    if method not in ("dmtet", "fc"):
        raise ValueError(f"unknown isosurface method {method!r} (expected 'dmtet' or 'fc')")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)

    # mc16's grid box, then an isotropic normalisation so the SDF, the
    # deformation bound and the learning rate live in one unit
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    lo, hi = V.min(0), V.max(0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    center, L = 0.5 * (lo + hi), float((hi - lo).max())

    if method == "fc":
        fc = FlexiCubes(device=str(dev))
        unit, cells = fc.construct_voxel_grid(n - 1)  # (n-1)^3 cubes, n^3 vertices
        unit = unit.double().cpu().numpy()
        edge_pairs = [0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4]
    else:
        unit, cells = kuhn_tet_grid(n)
        cells = torch.as_tensor(cells, dtype=torch.long, device=dev)
        edge_pairs = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
    grid = lo + (unit + 0.5) * (hi - lo)  # world, per-axis box like mc's linspace

    # initial SDF and mc's surface snapping: a grid point within half a cell
    # (per axis) of the surface moves onto it with SDF -1e-5, which keeps a
    # zero crossing next to features thinner than a cell
    S, _, C, _ = igl.signed_distance(grid, V, F)
    half = (hi - lo) / n * 0.5  # exactly mc's threshold (its res is the point count too)
    snap = np.all(np.abs(C - grid) < half, axis=1)
    grid[snap] = C[snap]
    S = np.where(snap, -1e-5, S)

    x0 = torch.as_tensor((grid - center) / L, dtype=torch.float32, device=dev)
    sdf = torch.nn.Parameter(torch.as_tensor(S / L, dtype=torch.float32, device=dev))
    deform = torch.nn.Parameter(torch.zeros_like(x0))
    edges = _unique_edges(cells, edge_pairs)
    max_disp = (x0[edges[:, 0]] - x0[edges[:, 1]]).norm(dim=1).mean() / 4  # nvdiffrec's FlexiCubes bound

    if method == "fc":
        weights = torch.nn.Parameter(torch.ones(cells.shape[0], 21, device=dev))
        params = [sdf, deform, weights]

        def extract(training):
            v, f, l_dev = fc(x0 + max_disp * torch.tanh(deform), sdf, cells, n - 1,
                             weights[:, :12], weights[:, 12:20], weights[:, 20], training=training)
            return v, f, l_dev.mean() if l_dev.numel() else torch.zeros((), device=dev)
    else:
        mt = DMTet(device=dev)
        params = [sdf, deform]

        def extract(training):
            v, f = mt(x0 + max_disp * torch.tanh(deform), -sdf, cells)  # DMTet: inside > 0
            return v, f, torch.zeros((), device=dev)

    if fit_steps > 0:
        Vn = torch.as_tensor((V - center) / L, dtype=torch.float32, device=dev)
        Fn = torch.as_tensor(F, dtype=torch.long, device=dev)
        target = _sample_surface(Vn, Fn, n_samples, gen)
        opt = torch.optim.Adam(params, lr=lr)
        first = last = None
        for it in range(fit_steps):
            v, f, l_dev = extract(training=True)
            if f.shape[0] == 0:
                logger.warning("%s%d fit: surface vanished at step %d, keeping the last one", method, n, it)
                break
            cd = _chamfer(_sample_surface(v, f, n_samples, gen), target)
            # nvdiffrec's schedule: 0.2 decaying to 0.01 over the first quarter
            sdf_w = 0.2 - (0.2 - 0.01) * min(1.0, 4.0 * it / fit_steps)
            loss = cd + sdf_w * sdf_reg_loss(sdf, edges).mean() * 1e-3 + 0.25 * l_dev * 1e-3
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = float(cd) if first is None else first
            last = float(cd)
        if first is not None:
            logger.info("%s%d fit: chamfer %.3g -> %.3g (normalised units, %d steps)", method, n, first, last, fit_steps)

    with torch.no_grad():
        v, f, _ = extract(training=False)
    verts = v.double().cpu().numpy() * L + center
    faces = f.long().cpu().numpy()

    # wind like igl.marching_cubes on an inside-negative SDF, which is
    # inward-facing (convert_mesh_to_mc flips it afterwards): match on the
    # sign of the enclosed volume
    if faces.shape[0]:
        t = verts[faces]
        vol = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum()
        if vol > 0:
            faces = faces[:, [0, 2, 1]]
    return verts, faces
