import importlib

import pytest

from vl_saliency.selectors.indexing import AbsoluteIndex, IndexFromEnd

indexing = importlib.import_module("vl_saliency.selectors.indexing")

# ---------------- Absolute Index --------------------


def test_absolute_index_generated_token(dummy_trace):
    selector = AbsoluteIndex(3)  # Absolute token index 3
    dummy_trace.gen_start = 2  # Generation starts at token index 2
    token_index = selector(dummy_trace)

    assert token_index == 1  # gen_start is 2, so relative index is 3 - 2 = 1


def test_absolute_index_non_generated_token(dummy_trace):
    selector = AbsoluteIndex(1)  # Absolute token index 1
    dummy_trace.gen_start = 2  # Generation starts at token index 2

    with pytest.raises(ValueError, match="non-generated"):
        selector(dummy_trace)


def test_absolute_index_no_gen_start_warning(dummy_trace, monkeypatch, caplog):
    monkeypatch.setattr(indexing.logger, "propagate", True)

    selector = AbsoluteIndex(4)  # Absolute token index 4
    dummy_trace.gen_start = None  # No generation start info

    with caplog.at_level("WARNING"):
        token_index = selector(dummy_trace)

    assert token_index == 4  # Should return absolute index as-is
    assert "no gen_start" in caplog.text


# ---------------- Index From End --------------------


def test_index_from_end_valid_offset(dummy_trace):
    selector = IndexFromEnd(2)  # 2nd token from the end
    dummy_trace.total_generated_tokens = 5  # Total generated tokens

    token_index = selector(dummy_trace)

    assert token_index == 3  # 5 - 2 = 3


def test_index_from_end_offset_exceeds_total(dummy_trace):
    selector = IndexFromEnd(6)  # Offset exceeds total generated tokens
    dummy_trace.total_generated_tokens = 5  # Total generated tokens

    with pytest.raises(ValueError, match="exceeds total"):
        selector(dummy_trace)


def test_index_from_end_non_positive_offset(dummy_trace):
    selector = IndexFromEnd(-1)  # Non-positive offset

    with pytest.raises(ValueError, match="must be positive"):
        selector(dummy_trace)
