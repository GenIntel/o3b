"""Vendored MeshFM inference subset (see README.md).

Upstream's package ``__init__`` re-exported its CLI helpers (mesh_io, viz,
inference); only the network and the surface sampler are vendored, so this one
exports nothing and the wrapper imports the submodules directly.
"""
