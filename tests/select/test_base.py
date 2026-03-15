import pytest

from vl_saliency.select.base import IndexSelector


class DummySelector(IndexSelector):
    def __init__(self, idx):
        self.idx = idx

    def select(self, view):
        return self.idx


class DummyView:
    def __init__(self, num_tokens=10):
        self.num_tokens = num_tokens
        self.batch_idx = 0
        self.image_idx = 0


@pytest.fixture
def dummy_view():
    return DummyView()


def test_index_selector_returns_index(dummy_view):
    sel = DummySelector(3)

    assert sel(dummy_view) == 3


def test_index_selector_out_of_bounds(dummy_view):
    for idx in [-5, 20]:
        sel = DummySelector(idx)
        with pytest.raises(IndexError):
            sel(dummy_view)


@pytest.mark.parametrize(
    "base,expr,expected",
    [
        (3, lambda s: s + 2, 5),
        (5, lambda s: s - 2, 3),
        (5, lambda s: s + 3 - 1, 7),
    ],
)
def test_offset_behavior(dummy_view, base, expr, expected):
    sel = expr(DummySelector(base))
    assert sel(dummy_view) == expected
