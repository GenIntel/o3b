"""The table-driven Hilbert encoder must give encode_bitwise's codes exactly:
every cell of every cube up to depth 6, random points up to depth 16."""
import pytest
import torch

from o3b.model.litept.serialization.hilbert import encode, encode_bitwise

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth", range(1, 17))
def test_encode_equals_bitwise(depth, device):
    if depth <= 6:
        r = torch.arange(1 << depth)
        p = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    else:
        p = torch.randint(0, 1 << depth, (50000, 3), generator=torch.Generator().manual_seed(depth))
    p = p.int().to(device)       # LitePT passes int32 grid coordinates
    assert torch.equal(encode(p, 3, depth), encode_bitwise(p, 3, depth))


def test_single_point_shape():
    # encode_bitwise squeezes a single point to a 0-d tensor; both broadcast the same
    p = torch.tensor([[3, 1, 2]], dtype=torch.int32)
    assert encode(p, 3, 4).view(-1).tolist() == encode_bitwise(p, 3, 4).view(-1).tolist()
