from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from vl_saliency.core.map import SaliencyMap


@pytest.fixture
def smap() -> SaliencyMap:
    t = torch.arange(0, 16.0).view(1, 1, 4, 4)
    return SaliencyMap(t)


# ------------------------- Initialization -------------------------


def test_constructor_requires_4d_tensor():
    with pytest.raises(ValueError):
        SaliencyMap(torch.zeros(3, 3, 3))


def test_equality_and_clone_independence(smap: SaliencyMap):
    clone = SaliencyMap(smap.tensor())
    assert smap == clone

    assert smap != smap.map  # different types

    # changing returned tensor doesn't mutate original (clone expected)
    t = smap.tensor()
    t += 1
    assert smap == clone  # original unchanged


def test_conversion_save(smap: SaliencyMap, tmp_path, monkeypatch):
    np_array = smap.numpy()
    assert isinstance(np_array, np.ndarray)
    assert np_array.shape == (1, 1, 4, 4)

    captured = {}
    monkeypatch.setattr(
        torch, "save", lambda obj, path, **kwargs: captured.update({"obj": obj, "path": path, "kwargs": kwargs})
    )

    save_path = tmp_path / "smap.pt"
    smap.save(str(save_path))
    assert captured["path"] == str(save_path)
    assert torch.equal(captured["obj"], smap.map)


# ------------------------- Arithmetic Operations -------------------------


def test_add_sub(smap: SaliencyMap):
    result = smap + smap
    expected = smap.tensor() * 2
    assert torch.equal(result.tensor(), expected)

    result = smap - smap
    expected = torch.zeros_like(smap.tensor())
    assert torch.equal(result.tensor(), expected)


def test_mul_div(smap: SaliencyMap):
    result = smap * 2
    expected = smap.tensor() * 2
    assert torch.equal(result.tensor(), expected)

    result = smap / 2
    expected = smap.tensor() / 2
    assert torch.equal(result.tensor(), expected)


# ------------------------- Torch Function Support -------------------------


def test_torch_function_add(smap: SaliencyMap):
    result = torch.add(smap, smap)  # type: ignore[arg-type]
    expected = smap.tensor() * 2
    assert torch.equal(result.tensor(), expected)  # type: ignore[arg-type]


def test_torch_function_unsupported(smap: SaliencyMap):
    with pytest.raises(NotImplementedError):
        torch.fft.fft2(smap)  # type: ignore[arg-type]


def test_torch_function_inplace_out(smap: SaliencyMap):
    with pytest.raises(TypeError):
        torch.add(smap, smap, out=smap.tensor())  # type: ignore[arg-type]


# ------------------------- Transform Application -------------------------


def test_apply_transform(smap: SaliencyMap):
    def dummy_transform(map: SaliencyMap) -> SaliencyMap:
        new_tensor = map.tensor() + 10
        return SaliencyMap(new_tensor)

    result = smap.apply(dummy_transform)
    expected = smap.tensor() + 10
    assert torch.equal(result.tensor(), expected)

    result2 = smap >> dummy_transform
    assert torch.equal(result2.tensor(), expected)


def test_apply_transform_inplace(smap: SaliencyMap):
    def inplace_transform(map: SaliencyMap) -> SaliencyMap:
        map.map += 5  # Attempt to modify in place
        return map

    new_smap = smap.apply(inplace_transform)
    assert smap != new_smap  # Original should remain unchanged


def test_agg(smap: SaliencyMap):  # More tests in transforms/test_layers.py
    agg_map = smap.agg()
    expected = smap.tensor().mean(dim=(0, 1), keepdim=True)
    assert torch.equal(agg_map.tensor(), expected)


def test_plot(smap: SaliencyMap, monkeypatch):
    mock_show = MagicMock()
    monkeypatch.setattr(Figure, "show", mock_show)

    # Basic test to ensure plot runs without error
    smap.agg().plot(show=False)
    assert mock_show.call_count == 0

    # Test show=True (will display plot)
    smap.agg().plot(show=True)
    assert mock_show.call_count == 1

    smap2 = SaliencyMap(torch.rand(2, 3, 8, 8))
    with pytest.raises(ValueError):
        smap2.plot(show=False)  # Invalid image shape
