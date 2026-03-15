import pytest

from vl_saliency.select.factories import absolute, from_end
from vl_saliency.select.pos import AbsoluteIndex, ReverseIndex


def test_absolute_index_selection(dummy_saliency_grid):
    scoped = dummy_saliency_grid.view(batch_idx=1, image_idx=0)

    # Gen Indices: [1, 2]
    # All indices: range(3)
    # Absolute index 1 should select the first gen token (index 0 in gen tokens)
    selector = AbsoluteIndex(index=1)
    selected_idx = selector(scoped)
    assert selected_idx == 0

    # Absolute index 2 should select the second gen token (index 1 in gen tokens)
    selector = AbsoluteIndex(index=2)
    selected_idx = selector(scoped)
    assert selected_idx == 1

    # Ensure that selecting an out-of-bounds index raises an error
    with pytest.raises(IndexError):
        selector = AbsoluteIndex(index=3)  # Out of bounds
        selector(scoped)


def test_reverse_index_selection(dummy_saliency_grid):
    scoped = dummy_saliency_grid.view(batch_idx=1, image_idx=0)

    # Gen Indices: [1, 2]
    # Reverse index 0 should select the last gen token (index 1 in gen tokens)
    selector = ReverseIndex(offset_from_end=0)
    selected_idx = selector(scoped)
    assert selected_idx == 1

    # Reverse index 1 should select the first gen token (index 0 in gen tokens)
    selector = ReverseIndex(offset_from_end=1)
    selected_idx = selector(scoped)
    assert selected_idx == 0

    # Ensure that you cannot create a ReverseIndex with a negative offset
    with pytest.raises(IndexError):
        selector = ReverseIndex(offset_from_end=-1)  # Out of bounds
        selector(scoped)

    # Ensure that selecting an out-of-bounds reverse index raises an error
    with pytest.raises(IndexError):
        selector = ReverseIndex(offset_from_end=2)  # Out of bounds
        selector(scoped)


def test_factories():
    idx = absolute(5)

    assert isinstance(idx, AbsoluteIndex)
    assert idx.index == 5

    assert repr(idx) == repr(AbsoluteIndex(5))

    idx = from_end(2)

    assert isinstance(idx, ReverseIndex)
    assert idx.offset_from_end == 2

    assert repr(idx) == repr(ReverseIndex(2))


def test_pos_repr():
    absolute = AbsoluteIndex(5)
    assert "index=5" in repr(absolute)

    reverse = ReverseIndex(5)
    assert "offset_from_end=5" in repr(reverse)
