from typing import List
import cv2
import numpy as np
import torch


def get_sub_dims(dims: List[int], dims_banned: List[int]):
    sub_dims = [
        dim for dim in torch.arange(len(dims)).tolist() if dim not in dims_banned
    ]
    return sub_dims


def get_sub_shape(shape: torch.Size, dims_banned: List[int]):
    sub_shape = [
        shape[dim]
        for dim in torch.arange(len(shape)).tolist()
        if dim not in dims_banned
    ]
    return torch.Size(sub_shape)


def mask_from_pxl2d(pxl2d: torch.Tensor, dim_pxl: int, dim_pts: int, H: int, W: int):
    """
    Args:
        pxl2d (torch.Tensor): ...x2x... 2d pixel information
        dim_pxl (int): Dimension of the 2D pixel information
        dim_pts (int): Dimension of the points belonging to one mask
        H (int): Output height.
        W (int): Output width.

    Returns:
        mask (torch.Tensor): ..xHxW mask which contains ones at 2d pixels.
    """

    device = pxl2d.device
    shape_mask = get_sub_shape(
        pxl2d.shape,
        dims_banned=[dim_pxl, dim_pts],
    ) + torch.Size([H, W])
    mask = torch.zeros(size=shape_mask, dtype=torch.bool, device=device)

    pxl2d_x = torch.index_select(
        pxl2d,
        index=torch.LongTensor([0]).to(device=device),
        dim=dim_pxl,
    ).to(dtype=torch.long)[..., 0]
    pxl2d_y = torch.index_select(
        pxl2d,
        index=torch.LongTensor([1]).to(device=device),
        dim=dim_pxl,
    ).to(dtype=torch.long)[..., 0]

    pxl2d_has_zero = (
        (pxl2d == 0).all(dim=dim_pxl, keepdim=True).all(dim=dim_pts, keepdim=True)
    )
    pxl2d_has_zero = pxl2d_has_zero.squeeze(dim=(dim_pts, dim_pxl))

    pxl2d_mask_out_of_bounds = (
        (pxl2d_x < 0) + (pxl2d_x > W - 1) + (pxl2d_y < 0) + (pxl2d_y > H - 1)
    )
    pxl1d = pxl2d_y * W + pxl2d_x
    pxl1d[pxl2d_mask_out_of_bounds] = 0
    mask = mask.reshape(*mask.shape[:-2], -1)
    mask = torch.scatter(
        dim=2,
        input=mask,
        index=pxl1d.permute(1, 2, 0),
        src=torch.ones(size=pxl1d.permute(1, 2, 0).shape, dtype=torch.bool).to(
            device=device,
        ),
    )

    mask = mask.reshape(*mask.shape[:-1], H, W)

    mask[:, :, 0, 0] = pxl2d_has_zero

    return mask


def _mask_distance_transform_cv2(mask_bin):
    """Exact euclidean DT, one cv2 call per image on the CPU."""
    device = mask_bin.device
    mask_bin = mask_bin.cpu() * 1.0
    mask_dt = []
    for _mask_bin in mask_bin:
        mask_np = np.uint8(_mask_bin.numpy() * 255.0)
        _mask_dt = torch.FloatTensor(
            cv2.distanceTransform(mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
        )[
            None,
        ]
        mask_dt.append(_mask_dt)
    return torch.cat(mask_dt, dim=0).to(device)


def _sq_edt_1d(f, chunk_bytes):
    """``out[n, r, i] = min_j ((i - j)^2 + f[n, r, j])`` for f of shape (N, R, L).

    The lower envelope by brute force: O(L) work per output element, but a
    single batched ``amin`` — which is what makes it fast on a GPU, where the
    O(L) sequential scan of Felzenszwalb's algorithm is worth little. The
    (N, R, i-chunk, L) intermediate is chunked along i to stay within
    ``chunk_bytes``.
    """
    N, R, L = f.shape
    idx = torch.arange(L, device=f.device, dtype=f.dtype)
    row_bytes = max(1, N * R * L * f.element_size())
    step = max(1, int(chunk_bytes // row_bytes))
    outs = []
    for s in range(0, L, step):
        d2 = (idx[s:s + step].view(-1, 1) - idx.view(1, -1)) ** 2  # (chunk, L)
        outs.append((f.unsqueeze(2) + d2.view(1, 1, *d2.shape)).amin(dim=-1))
    return torch.cat(outs, dim=2)


def _mask_distance_transform_torch(mask_bin, chunk_bytes=int(128 * 2**20)):
    """Exact euclidean DT on the input's own device (no CPU round trip).

    Separable squared-EDT: a 1-D transform down the columns, then one across
    the rows (Felzenszwalb & Huttenlocher's decomposition, with the 1-D step
    brute-forced — see :func:`_sq_edt_1d`). Matches cv2 to float32 precision,
    including the truncated case: pixels outside the image are *not* treated as
    background, so a mask running off the image edge keeps growing inwards.

    A mask with no background pixel at all has no defined distance; it comes
    out as a huge value here (as it does from cv2, which returns ~1.8e19) and
    is overwritten by :func:`get_mask_distance_transform_norm`.
    """
    N, H, W = mask_bin.shape
    f = torch.where(
        mask_bin,
        torch.full((), 1e10, device=mask_bin.device, dtype=torch.float32),
        torch.zeros((), device=mask_bin.device, dtype=torch.float32),
    )
    # along y (columns), then along x (rows)
    g = _sq_edt_1d(f.permute(0, 2, 1).contiguous(), chunk_bytes).permute(0, 2, 1)
    d2 = _sq_edt_1d(g.contiguous(), chunk_bytes)
    return d2.clamp_min(0.0).sqrt()


def _mask_distance_transform_kornia(mask_bin, kernel_size=3, h=0.35):
    """Approximate DT via ``kornia.contrib.distance_transform`` (cascaded convs).

    Kept for comparison only — it is *not* an euclidean distance: the cascade
    accumulates a fixed ``kernel_size // 2`` offset per iteration, which makes
    the result chessboard-like (exact along the axes, ~28% short along the
    diagonals, i.e. the full Chebyshev-vs-euclidean gap). On this repo's
    silhouettes that biases the ``mask_dt`` loss term by ~20%, and it is slower
    than the exact ``torch`` backend anyway (30 masks on an RTX 3090: 64x64
    6.1 ms, 128x128 24.9 ms — the cascade needs one 3x3 convolution per pixel of
    propagated distance). ``kornia`` is an optional dependency, imported here
    only when this backend is asked for. See ``backend`` in
    :func:`get_mask_distance_transform`.
    """
    import kornia

    # kornia measures the distance to the nearest *non-zero* pixel, so the
    # sources are the background pixels
    return kornia.contrib.distance_transform(
        (~mask_bin).float().unsqueeze(1), kernel_size=kernel_size, h=h,
    ).squeeze(1)


def get_mask_distance_transform(mask_bin, backend="auto"):
    # mask_bin: ...xHxW: bool
    # mask_dt: ...xHxW : distance in pixels to the edge of the mask,
    #                    note: 0 means outside of mask,
    """Distance (in pixels) from each mask pixel to the nearest background one.

    Args:
        backend: which implementation computes it.
            - ``"cv2"``: exact, CPU only (``cv2.distanceTransform``).
            - ``"torch"``: exact, runs wherever the input is — matches cv2 to
              float32 precision and avoids the GPU→CPU→GPU round trip, which
              makes it several times faster than ``"cv2"`` for live losses on a
              CUDA tensor (RTX 3090, 30 masks: 64x64 0.8 vs 6.4 ms, 128x128 1.5
              vs 11.1 ms, 256x256 12.4 vs 38.2 ms), and ~15x slower than cv2 on
              the CPU.
            - ``"kornia"``: approximate and chessboard-biased, see
              :func:`_mask_distance_transform_kornia`.
            - ``"auto"`` (default): ``"torch"`` on CUDA, ``"cv2"`` on the CPU —
              the fastest exact option on either device.
    """
    mask_in_shape = mask_bin.shape
    mask_bin = mask_bin.reshape(-1, mask_bin.shape[-2], mask_bin.shape[-1])

    if backend == "auto":
        backend = "torch" if mask_bin.is_cuda else "cv2"
    if backend == "cv2":
        mask_dt = _mask_distance_transform_cv2(mask_bin)
    elif backend == "torch":
        mask_dt = _mask_distance_transform_torch(mask_bin.bool())
    elif backend == "kornia":
        mask_dt = _mask_distance_transform_kornia(mask_bin.bool())
    else:
        raise ValueError(
            f"unknown distance transform backend '{backend}', "
            f"expected one of 'auto', 'cv2', 'torch', 'kornia'",
        )

    return mask_dt.reshape(*mask_in_shape)


def get_mask_distance_transform_norm(mask_bin, backend="auto"):
    # mask_bin: ...xHxW: bool
    # mask_dt: ...xHxW : distance in pixels normalized [0., 1.] to the edge of the mask,
    #                    note: 0 means outside of mask,

    mask_dt = get_mask_distance_transform(mask_bin, backend=backend)
    mask_size = max(mask_dt.shape[-1], mask_dt.shape[-2])
    mask_dt /= mask_size

    mask_dt[(~mask_bin).sum(dim=-2).sum(dim=-1) == 0] = 1.

    return mask_dt