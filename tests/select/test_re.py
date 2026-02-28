import pytest
import torch

from vl_saliency.core.scoped import ScopedSaliencyGrid
from vl_saliency.select import ReSelector

# ------- Fixtures -------


@pytest.fixture
def dummy_scoped_grid(dummy_saliency_grid, dummy_tokenizer):
    return ScopedSaliencyGrid(
        dummy_saliency_grid,
        batch_idx=1,
        image_idx=0,
        input_ids=torch.tensor([0, 1, 2]),
        processor=dummy_tokenizer,
    )


# ------- Tests -------


def test_re_selector(dummy_saliency_grid, dummy_scoped_grid):
    selector = ReSelector(pattern=r"any", require_exact_match=False, select="first")

    grid = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0)
    with pytest.raises(ValueError):
        selector(grid)  # No input ids or tokenizer set, should raise ValueError

    selector = ReSelector(pattern="token_0", require_exact_match=True, select="first")
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # token_0 is an image token, won't match

    selector = ReSelector(pattern="token_1", require_exact_match=True, select="first")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 0


def test_re_selector_no_match(dummy_scoped_grid):
    selector = ReSelector(pattern="nonexistent_token", require_exact_match=True, select="first")
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # No tokens match the pattern, should raise ValueError


def test_re_selector_exact_match(dummy_scoped_grid):
    selector = ReSelector(pattern="token", require_exact_match=True, select="first")
    with pytest.raises(ValueError):
        selector(dummy_scoped_grid)  # No tokens match the pattern exactly, should raise ValueError

    selector = ReSelector(pattern="^token_1$", require_exact_match=True, select="first")
    selected_indices = selector(dummy_scoped_grid)  # no double anchoring should occur
    assert selected_indices == 0


def test_re_selector_select_partial(dummy_scoped_grid):
    selector = ReSelector(pattern="token", require_exact_match=False, select="first")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 0

    selector = ReSelector(pattern="token", require_exact_match=False, select="last")
    selected_indices = selector(dummy_scoped_grid)
    assert selected_indices == 1
