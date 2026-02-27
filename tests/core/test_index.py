import pytest

from vl_saliency.core.index import Index


def test_index_from_indices():
    # Test with an Index instance
    idx1 = Index(batch_idx=0, image_idx=1, token_idx=2)
    assert Index.from_indices(idx1) == idx1

    # Test with a single integer (token index)
    idx2 = 5
    expected2 = Index(batch_idx=None, image_idx=None, token_idx=5)
    assert Index.from_indices(idx2) == expected2

    # Test with a tuple of one integer (token index)
    idx3 = (7,)
    expected3 = Index(batch_idx=None, image_idx=None, token_idx=7)
    assert Index.from_indices(idx3) == expected3

    # Test with a tuple of two integers (img index, token index)
    idx4 = (1, 3)
    expected4 = Index(batch_idx=None, image_idx=1, token_idx=3)
    assert Index.from_indices(idx4) == expected4

    # Test with a tuple of three integers (batch index, img index, token index)
    idx5 = (0, 2, 4)
    expected5 = Index(batch_idx=0, image_idx=2, token_idx=4)
    assert Index.from_indices(idx5) == expected5

    # Test with an invalid tuple (too many elements)
    idx6 = (1, 2, 3, 4)
    with pytest.raises(IndexError):
        Index.from_indices(idx6)  # Too many elements in the tuple

    # Test None values are removed and padded
    idx7 = (None, 1, None)
    expected7 = Index(batch_idx=None, image_idx=None, token_idx=1)
    assert Index.from_indices(idx7) == expected7 == Index.from_indices((1,))
