"""Differentiable weighted Umeyama (similarity) fit, batched, torch only.

``weighted_umeyama(src, dst, w)`` gives ``s, R, t`` with ``dst ≈ s R src + t``
(least squares, weights ``w``). The rotation is the Procrustes solution
``R = U diag(1, 1, det(U V^T)) V^T`` of the weighted cross-covariance, and its
backward is written out (``_ProcrustesRot``): autograd through
``torch.linalg.svd`` divides by *differences* of singular values and returns inf
/ NaN whenever two are equal, which is the normal case for a rotationally
symmetric object's NOCS (a bottle, a bowl). The rotation factor itself is smooth
there; its derivative only divides by *sums* of the (sign-corrected) singular
values, ``dR = U' Omega V^T`` with
``Omega_ij = (P_ij - P_ji) / (s_i + s_j)``, ``P = U'^T dM V``, which fails only
for a cross-covariance of rank <= 1 (all points on a line). Those sums are
clamped away from zero.
"""
import torch


class _ProcrustesRot(torch.autograd.Function):
    """R = argmax_R tr(R^T M) over rotations, for M (B, 3, 3); float64 inside."""

    @staticmethod
    def forward(ctx, M):
        Md = M.double()
        U, S, Vh = torch.linalg.svd(Md)
        d = torch.sign(torch.linalg.det(U @ Vh))
        d = torch.where(d == 0, torch.ones_like(d), d)
        sign = torch.ones_like(S)
        sign[:, 2] = d
        Us = U * sign[:, None, :]                 # U' = U diag(1, 1, d)
        R = Us @ Vh
        ctx.save_for_backward(Us, S * sign, Vh.transpose(1, 2))
        return R.to(M.dtype)

    @staticmethod
    def backward(ctx, gR):
        Us, s, V = ctx.saved_tensors
        A = Us.transpose(1, 2) @ gR.double() @ V
        den = s[:, :, None] + s[:, None, :]
        eps = 1e-6 * s[:, :1, None].abs().clamp_min(1e-12)
        sgn = torch.where(den < 0, -torch.ones_like(den), torch.ones_like(den))
        den = sgn * den.abs().clamp_min(eps)                              # |s_i + s_j| >= eps
        gP = (A - A.transpose(1, 2)) / den
        gM = Us @ gP @ V.transpose(1, 2)
        return gM.to(gR.dtype)


def procrustes_rot3x3(M: torch.Tensor) -> torch.Tensor:
    """The rotation closest to ``M`` (B, 3, 3) in ``tr(R^T M)``, with a stable gradient."""
    return _ProcrustesRot.apply(M)


def weighted_umeyama(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor = None, eps: float = 1e-12):
    """Least-squares similarity ``dst ≈ s R src + t`` for (B, N, 3) point sets.

    ``w`` (B, N) are non-negative weights, normalised here (uniform if None).
    Returns ``s`` (B,), ``R`` (B, 3, 3), ``t`` (B, 3), differentiable in all three
    inputs. Computed in float64, returned in ``src``'s dtype.
    """
    dtype = src.dtype
    src, dst = src.double(), dst.double()
    if w is None:
        w = torch.ones(src.shape[:2], dtype=src.dtype, device=src.device)
    w = w.double()
    w = w / w.sum(1, keepdim=True).clamp_min(eps)
    mu_s = (w[..., None] * src).sum(1)
    mu_d = (w[..., None] * dst).sum(1)
    xs, xd = src - mu_s[:, None], dst - mu_d[:, None]
    M = torch.einsum("bn,bni,bnj->bij", w, xd, xs)        # weighted cross-covariance dst x src
    R = procrustes_rot3x3(M)
    var_s = (w * xs.square().sum(-1)).sum(1).clamp_min(eps)
    s = (R * M).sum((1, 2)) / var_s                          # tr(R^T M) / var(src)
    t = mu_d - s[:, None] * (R @ mu_s[..., None])[..., 0]
    return s.to(dtype), R.to(dtype), t.to(dtype)
