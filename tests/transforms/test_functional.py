import numpy as np
import torch

from vl_saliency.transforms.functional import abs, normalize, relu, sigmoid


def test_relu(smap):
    transformed = smap.apply(relu)

    expected = torch.relu(smap.tensor())
    assert torch.allclose(transformed.tensor(), expected)


def test_abs(smap):
    transformed = smap.apply(abs)
    expected = torch.abs(smap.tensor())
    assert torch.allclose(transformed.tensor(), expected)


def test_sigmoid(smap):
    transformed = smap.apply(sigmoid)
    expected = torch.sigmoid(smap.tensor())
    assert torch.allclose(transformed.tensor(), expected)


def test_normalize(smap):
    tensor = smap.tensor()
    transformed = smap.apply(normalize)

    map_np = tensor.detach().cpu().numpy()
    normalized = (map_np - map_np.min()) / (np.ptp(map_np) + 1e-8)
    expected = torch.from_numpy(normalized).to(device=tensor.device, dtype=tensor.dtype)
    assert torch.allclose(transformed.tensor(), expected)
