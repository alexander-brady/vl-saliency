import numpy as np
import pytest
import torch

from vl_saliency.core.map import SaliencyMap
from vl_saliency.transforms.localization import (
    LocalizationHeads,
    _elbow_chord,
    _spatial_entropy,
    identify_localization_heads,
)

# ------------------------------ Spatial Entropy ------------------------------ #


def test_spatial_entropy():
    # Test case 1: Uniform distribution (will not form components above threshold)
    attn_map = torch.ones((4, 4)) / 16  # 4x4 uniform attention map
    se = _spatial_entropy(attn_map)
    assert se == float("inf"), f"Expected entropy of inf, got {se}"

    # Test case 2: Concentrated distribution
    attn_map = torch.zeros((4, 4))
    attn_map[0, 0] = 1.0  # All attention on one pixel
    se = _spatial_entropy(attn_map)
    assert np.isclose(se, 0.0), f"Expected entropy of 0.0, got {se}"

    # Test case 3: No attention mass
    attn_map = torch.zeros((4, 4))  # All zeros
    se = _spatial_entropy(attn_map)
    assert se == float("inf"), f"Expected entropy of inf, got {se}"


# ------------------------------ Elbow Chord ------------------------------ #


def test_elbow_chord():
    values = [1, 2, 3, 5, 8, 13, 21.0]
    threshold = _elbow_chord(values)
    assert isinstance(threshold, float), "Threshold should be a float value"


def test_elbow_chord_few_values():
    # Test with 2 values
    values = [0.1, 0.2]
    threshold = _elbow_chord(values)
    assert threshold == 0.1, "Expected minimum value for 2 inputs"

    # Test with 1 value
    values = [42.0]
    threshold = _elbow_chord(values)
    assert threshold == 42.0, "Expected the single value for 1 input"

    # Test with empty list
    values = []
    threshold = _elbow_chord(values)
    assert threshold == 0.0, "Expected 0.0 for empty input"


# ------------------------------ Identify Localization Heads ------------------------------ #


def test_basic_selection_and_ordering_by_entropy():
    # 3 layers, 2 heads, 2x2 maps; pick from layer 2 (>1)
    t = torch.zeros(3, 2, 2, 2)
    # (2,0): concentrated attention -> lower spatial entropy (better)
    t[2, 0, 0, 0] = 0.3
    # (2,1): more spread -> higher spatial entropy
    t[2, 1, 0, 0] = 0.2
    t[2, 1, 0, 1] = 0.1

    out = identify_localization_heads(SaliencyMap(t), chord_thresholding=False, min_keep=1, max_keep=5)
    assert out == [(2, 0), (2, 1)]


def test_localization_head_selection_respects_chord_thresholding():
    t = torch.zeros(4, 2, 2, 2)
    # (2,0): high attention sum
    t[2, 0, 0, 0] = 0.5
    # (2,1): low attention sum
    t[2, 1, 0, 0] = 0.1
    # (3,0): medium attention sum
    t[3, 0, 0, 0] = 0.3
    # (3,1): low attention sum
    t[3, 1, 0, 0] = 0.05

    out = identify_localization_heads(SaliencyMap(t), chord_thresholding=True, min_keep=1, max_keep=5)
    assert out == [(2, 0), (2, 1), (3, 0)]


def test_respects_max_keep():
    t = torch.zeros(3, 2, 2, 2)
    # (2,0) more spread (worse entropy)
    t[2, 0, 0, 0] = 0.15
    t[2, 0, 0, 1] = 0.15
    # (2,1) concentrated (better entropy)
    t[2, 1, 0, 0] = 0.3

    out = identify_localization_heads(SaliencyMap(t), chord_thresholding=False, min_keep=1, max_keep=1)
    assert out == [(2, 1)]


def test_fallback_min_keep_uses_top_attn_sum():
    # Only layers 0–1 have attention -> none pass (layer > 1), so fallback by attn sum
    t = torch.zeros(2, 3, 2, 2)
    t[0, 0, 0, 0] = 0.1  # sum 0.1
    t[0, 1, 0, 0] = 0.4  # sum 0.4  (top)
    t[1, 2, 0, 0] = 0.2  # sum 0.2  (second)

    out = identify_localization_heads(SaliencyMap(t), chord_thresholding=False, min_keep=2, max_keep=None)
    assert set(out) == {(0, 1), (1, 2)}


def test_skips_heads_attending_bottom_row():
    t = torch.zeros(3, 2, 2, 2)
    # good: layer 2, head 0 (no bottom-row attention)
    t[2, 0, 0, 0] = 0.3
    # bad: layer 2, head 1 attends bottom row (>0.05)
    t[2, 1, -1, 0] = 0.06

    out = identify_localization_heads(SaliencyMap(t), chord_thresholding=False, min_keep=1, max_keep=5)
    assert out == [(2, 0)]


def test_raises_when_no_heads_and_min_keep_zero():
    # All in layer 0, so nothing satisfies layer>1 and fallback disabled by min_keep=0
    t = torch.zeros(1, 2, 2, 2)
    t[0, 0, 0, 0] = 0.01
    t[0, 1, 0, 0] = 0.02

    with pytest.raises(ValueError, match="No heads were selected"):
        identify_localization_heads(SaliencyMap(t), chord_thresholding=False, min_keep=0, max_keep=None)


# ------------------------------ Transform ------------------------------- #


def test_localization_transform():
    t = torch.zeros(3, 2, 2, 2)
    t[2, 0, 0, 0] = 0.3

    smap = SaliencyMap(t)

    params = {
        "chord_thresholding": False,
        "min_keep": 1,
        "max_keep": 5,
    }

    out = identify_localization_heads(smap, **params)
    assert out == [(2, 0)]

    tensor = smap.tensor()[[2], [0]].unsqueeze(0)  # shape: [1, 1, H, W]
    selector = LocalizationHeads(**params)
    transformed = smap >> selector  # shape: [1, 1, H, W]
    assert torch.equal(transformed.tensor(), tensor)
