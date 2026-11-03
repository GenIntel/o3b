"""The caches added for Trinity-Frame's batch-1 latency must not change results."""
import pytest
import torch

from o3b.cv.geometry.transform import depth2pts3d_grid, reproj2d3d_broadcast

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def depth2pts3d_grid_uncached(depth, cam_intr4x4, opengl=False):
    H, W = depth.shape[-2:]
    pxl2d = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy"), dim=-1
                        ).to(device=cam_intr4x4.device, dtype=cam_intr4x4.dtype)
    homog = reproj2d3d_broadcast(pxl2d, proj4x4_inv=cam_intr4x4[..., None, None, :, :].inverse()
                                 ).transpose(-2, -1).transpose(-3, -2)
    first = torch.broadcast_shapes(depth.shape[:-3], cam_intr4x4.shape[:-2])
    pts3d = homog.expand(*first, 3, H, W) * depth.expand(*first, 1, H, W)
    if opengl:
        pts3d = pts3d * pts3d.new_tensor([1., -1., -1.]).view(*([1] * (pts3d.dim() - 3)), 3, 1, 1)
    return pts3d


@pytest.mark.parametrize("device", DEVICES)
def test_depth2pts3d_grid_cached_grid(device):
    g = torch.Generator().manual_seed(0)
    for B, H, W in ((1, 256, 256), (3, 24, 40), (2, 256, 256)):
        depth = torch.rand(B, 1, H, W, generator=g).to(device)
        K = torch.eye(4).repeat(B, 1, 1)
        K[:, 0, 0], K[:, 1, 1] = 300 + 50 * torch.rand(B, generator=g), 310.
        K[:, 0, 2], K[:, 1, 2] = W / 2, H / 2
        K = K.to(device)
        for gl in (False, True):
            assert torch.equal(depth2pts3d_grid(depth, K, gl), depth2pts3d_grid_uncached(depth, K, gl))


def test_dinov2_pos_encoding_cache():
    from o3b.model.dino.model import DINOv2
    try:
        m = DINOv2(hub_model="dinov2_vits14", freeze=True)
    except Exception as e:      # no hub cache / network
        pytest.skip(f"dinov2 not loadable: {e}")
    ext = m.extractor
    x = torch.zeros(1, 1 + 18 * 18, ext.embed_dim)
    with torch.no_grad():
        a, b = ext.interpolate_pos_encoding(x, 252, 252), ext.interpolate_pos_encoding(x, 252, 252)
        assert a is b                                    # cached
        fresh = type(ext).interpolate_pos_encoding(ext, x, 252, 252)
        assert torch.equal(a, fresh)
        ext.pos_embed.add_(1.)                           # an optimizer step bumps _version
        c = ext.interpolate_pos_encoding(x, 252, 252)
        assert not torch.equal(a, c)
        assert torch.equal(c, type(ext).interpolate_pos_encoding(ext, x, 252, 252))
    ext.pos_embed.requires_grad_(True)                   # trained: never cached, keeps its graph
    d = ext.interpolate_pos_encoding(x, 252, 252)
    assert d.grad_fn is not None
