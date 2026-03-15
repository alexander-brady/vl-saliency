import pytest
import torch

from vl_saliency.maps.view import SaliencyView
from vl_saliency.select.factories import regex
from vl_saliency.select.regex import RegexSelector

# ------- Fixtures -------


@pytest.fixture
def dummy_scoped_grid(dummy_saliency_grid, dummy_tokenizer):
    return SaliencyView(
        dummy_saliency_grid,
        batch_idx=1,
        image_idx=0,
        input_ids=torch.tensor([0, 1, 2]),
        processor=dummy_tokenizer,
    )


# ------- Tests -------


def test_re_selector(dummy_saliency_grid, dummy_scoped_grid):
    selector = RegexSelector(pattern=r"any", require_exact_match=False, occurrence="first")

    grid = SaliencyView(dummy_saliency_grid, batch_idx=1, image_idx=0)
    with pytest.raises(ValueError):
        selector(grid)  # No input ids or tokenizer set, should raise ValueError

    selector = RegexSelector(pattern="token_0", require_exact_match=True, occurrence="first")
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # token_0 is an image token, won't match

    selector = RegexSelector(pattern="token_1", require_exact_match=True, occurrence="first")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 0


def test_re_selector_no_match(dummy_scoped_grid):
    selector = RegexSelector(
        pattern="nonexistent_token", require_exact_match=True, occurrence="first"
    )
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # No tokens match the pattern, should raise ValueError


def test_re_selector_exact_match(dummy_scoped_grid):
    selector = RegexSelector(pattern="token", require_exact_match=True, occurrence="first")
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # No tokens match the pattern exactly, should raise ValueError

    selector = RegexSelector(pattern="^token_1$", require_exact_match=True, occurrence="first")
    selected_indices = selector(dummy_scoped_grid)  # no double anchoring should occur
    assert selected_indices == 0


def test_re_selector_select_partial(dummy_scoped_grid):
    selector = RegexSelector(pattern="token", require_exact_match=False, occurrence="first")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 0

    selector = RegexSelector(pattern="token", require_exact_match=False, occurrence="last")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 1


def test_regex_factory():
    sel = regex(r"\d+", occurrence="last")

    assert isinstance(sel, RegexSelector)
    assert sel.pattern.pattern == r"\d+"
    assert sel.occurrence == "last"

    assert repr(sel) == repr(RegexSelector(r"\d+", occurrence="last"))


def test_regex_selector_repr():
    sel = RegexSelector(r"\d+", occurrence="last")
    r = repr(sel)

    assert "RegexSelector(" in r
    assert r"\d+" in r
    assert "occurrence=last" in r
