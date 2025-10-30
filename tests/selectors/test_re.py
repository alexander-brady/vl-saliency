import pytest
import torch

from vl_saliency.selectors.re import ReSelector


class DummyTokenizer:
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        token_map = {
            0: "[CLS]",
            1: "Hello",
            2: "world",
            3: "!",
            4: "[SEP]",
        }

        return [token_map.get(i, "[UNK]") for i in ids]


@pytest.fixture
def dummier_trace(dummy_trace, dummy_processor):
    dummy_processor.tokenizer = DummyTokenizer()
    dummy_trace.processor = dummy_processor
    dummy_trace.gen_start = 0
    dummy_trace.token_ids = [0, 1, 2, 3, 4]  # [CLS] Hello world ! [SEP]
    return dummy_trace


# ---------------------------- Error Cases ----------------------------


def test_re_selector_no_processor(dummy_trace):
    selector = ReSelector(pattern="Hello")

    dummy_trace.processor = None  # Simulate missing processor
    with pytest.raises(ValueError, match="no processor"):
        selector(dummy_trace)


# ---------------------------- Functionality Tests ----------------------------


def test_re_selector_first_match(dummier_trace):
    selector = ReSelector(pattern="Hello", select="first", require_exact_match=False)
    index = selector(dummier_trace)
    assert index == 1  # "Hello" is at index 1 of generated tokens


def test_re_selector_last_match(dummier_trace):
    selector = ReSelector(pattern="world", select="last", require_exact_match=False)

    index = selector(dummier_trace)
    assert index == 2  # Last "world" is at index 2 of generated tokens


def test_re_selector_no_match(dummier_trace):
    selector = ReSelector(pattern="Goodbye", require_exact_match=False)

    with pytest.raises(ValueError, match="No tokens match the given pattern"):
        selector(dummier_trace)


def test_re_selector_exact_match(dummier_trace):
    selector = ReSelector(pattern="Hello", require_exact_match=True)

    index = selector(dummier_trace)
    assert index == 1  # Exact match for "Hello" at index 1 of generated tokens

    selector = ReSelector(pattern="lo", require_exact_match=True)
    with pytest.raises(ValueError, match="No tokens match the given pattern"):
        selector(dummier_trace)


def test_re_selector_partial_match(dummier_trace):
    selector = ReSelector(pattern="lo", require_exact_match=False)

    index = selector(dummier_trace)
    assert index == 1  # "Hello" contains "lo" at index 1 of generated tokens


def test_re_no_match_if_not_generated(dummier_trace):
    selector = ReSelector(pattern="Hello", require_exact_match=False)

    dummier_trace.gen_start = 2  # Only consider tokens from index 2 onwards
    with pytest.raises(ValueError, match="No tokens match the given pattern"):
        selector(dummier_trace)


def test_re_generated_id_shapes(dummier_trace):
    # Test with 1D generated_ids
    dummier_trace.generated_ids = torch.tensor([0, 1, 2, 3, 4])  # [CLS] Hello world ! [SEP]
    selector = ReSelector(pattern="world", require_exact_match=False)
    index = selector(dummier_trace)
    assert index == 2  # "world" is at index 2 of generated tokens

    # Test with 2D generated_ids
    dummier_trace.generated_ids = torch.tensor([[0, 1, 2, 3, 4]])  # [CLS] Hello world ! [SEP]
    index = selector(dummier_trace)
    assert index == 2  # "world" is at index 2 of generated tokens
