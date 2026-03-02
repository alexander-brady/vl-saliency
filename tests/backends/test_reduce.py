import pytest
import torch

from vl_saliency.backends.reduce import _HEAD_REDUCE, _LAYER_REDUCE


@pytest.fixture
def sample():
    scores = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ]
    )  # [1,2,2,2]
    mask = torch.tensor([[[[True, False], [True, True]]]])  # [1,1,2,2]
    return scores, mask


@pytest.mark.parametrize("name", ["mean", "sum", "max", "min", "prod"])
def test_head_reduce_shapes(sample, name):
    scores, mask = sample
    out = _HEAD_REDUCE[name](scores, mask)
    assert out.shape == (1, 2, 2)


def test_head_reduce_mean(sample):
    scores, mask = sample
    out = _HEAD_REDUCE["mean"](scores, mask)

    # (0, 0): (1 + 5) / 2 = 3
    # (0, 1): both masked → 0
    # (1, 0): (3 + 7) / 2 = 5
    # (1, 1): (4 + 8) / 2 = 6
    expected = torch.tensor([[[3.0, 0.0], [5.0, 6.0]]])
    assert torch.allclose(out, expected)


def test_head_reduce_sum(sample):
    scores, mask = sample
    out = _HEAD_REDUCE["sum"](scores, mask)
    expected = torch.tensor([[[6.0, 0.0], [10.0, 12.0]]])
    assert torch.allclose(out, expected)


def test_head_reduce_max(sample):
    scores, mask = sample
    out = _HEAD_REDUCE["max"](scores, mask)
    expected = torch.tensor([[[5.0, torch.finfo(scores.dtype).min], [7.0, 8.0]]])
    assert torch.allclose(out, expected)


def test_head_reduce_min(sample):
    scores, mask = sample
    out = _HEAD_REDUCE["min"](scores, mask)
    expected = torch.tensor([[[1.0, torch.finfo(scores.dtype).max], [3.0, 4.0]]])
    assert torch.allclose(out, expected)


def test_head_reduce_prod(sample):
    scores, mask = sample
    out = _HEAD_REDUCE["prod"](scores, mask)
    # (1*5)=5 ; masked → 1 ; (3*7)=21 ; (4*8)=32
    expected = torch.tensor([[[5.0, 1.0], [21.0, 32.0]]])
    assert torch.allclose(out, expected)


@pytest.mark.parametrize(
    "name,a,b,expected",
    [
        ("sum", torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])),
        (
            "mean",
            torch.tensor([[1.0]]),
            torch.tensor([[2.0]]),
            torch.tensor([[3.0]]),
        ),  # Muse be same as sum
        ("max", torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[2.0]])),
        ("min", torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor([[1.0]])),
        ("prod", torch.tensor([[3.0]]), torch.tensor([[2.0]]), torch.tensor([[6.0]])),
    ],
)
def test_layer_reduce(name, a, b, expected):
    out = _LAYER_REDUCE[name](a, b)
    assert torch.equal(out, expected)
