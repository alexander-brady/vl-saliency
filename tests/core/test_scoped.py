import pytest
import torch

from vl_saliency.core.scoped import ScopedSaliencyGrid

from ..utils import ImageSpec

# ------- Fixtures -------


@pytest.fixture
def dummy_tokenizer():
    class DummyTokenizer:
        def convert_ids_to_tokens(self, ids):
            return [f"token_{id}" for id in ids]

    return DummyTokenizer()


@pytest.fixture
def dummy_processor(dummy_tokenizer):
    class DummyProcessor:
        tokenizer = dummy_tokenizer

    return DummyProcessor()


# ------ Scoping tests -------


def test_scoped_init(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )

    scoped = ScopedSaliencyGrid(grid, batch_idx=0, image_idx=0)

    assert scoped.batch_idx == 0
    assert scoped.image_idx == 0
    assert scoped.num_tokens == 3
    assert scoped.gen_start_idx == 4  # after 4 image tokens
    assert scoped.gen_end_idx == 7  # 3 gen tokens after start idx

    with pytest.raises(ValueError):
        _ = scoped.gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_input_tokens

    scoped_2 = grid.scope(batch_idx=0, image_idx=0)
    assert scoped_2.batch_idx == 0
    assert scoped_2.image_idx == 0
    assert scoped_2.num_tokens == 3
    assert scoped_2.gen_start_idx == 4
    assert scoped_2.gen_end_idx == 7

    assert torch.equal(scoped.maps, scoped_2.maps)
    assert torch.equal(scoped.map(0), scoped_2.map(0))


def test_scoped_input_ids(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])  # 4 image tokens + 3 gen tokens
    scoped = ScopedSaliencyGrid(grid, batch_idx=0, image_idx=0, input_ids=input_ids)

    assert scoped.input_ids is not None
    assert scoped.input_ids.shape == (7,)
    assert torch.equal(scoped.input_ids, input_ids[0])

    assert torch.equal(scoped.gen_tokens, input_ids[0, 4:7])

    with pytest.raises(ValueError):
        _ = scoped.decoded_gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_input_tokens

    scoped_2 = grid.scope(batch_idx=0, image_idx=0, input_ids=input_ids[0])
    assert torch.equal(scoped_2.input_ids, scoped.input_ids)
    assert torch.equal(scoped_2.gen_tokens, scoped.gen_tokens)


def test_scoped_processor_tokenizer(build_sample_grid, dummy_processor, dummy_tokenizer):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])  # 4 image tokens + 3 gen tokens
    scoped = ScopedSaliencyGrid(
        grid, batch_idx=0, image_idx=0, input_ids=input_ids, processor=dummy_processor
    )

    assert scoped._tok is not None

    assert torch.equal(scoped.gen_tokens, input_ids[0, 4:7])
    assert scoped.decoded_gen_tokens == ["token_14", "token_15", "token_16"]
    assert scoped.decoded_input_tokens == [
        "token_10",
        "token_11",
        "token_12",
        "token_13",
        "token_14",
        "token_15",
        "token_16",
    ]

    scoped_2 = grid.scope(
        batch_idx=0, image_idx=0, input_ids=input_ids[0], processor=dummy_processor
    )
    assert torch.equal(scoped_2.gen_tokens, scoped.gen_tokens)
    assert scoped_2.decoded_gen_tokens == scoped.decoded_gen_tokens
    assert scoped_2.decoded_input_tokens == scoped.decoded_input_tokens


def test_scoped_selector(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16]])  # 4 image tokens + 3 gen tokens
    scoped = ScopedSaliencyGrid(grid, batch_idx=0, image_idx=0, input_ids=input_ids)

    def selector(scoped):
        return 0

    assert torch.equal(scoped.map(selector), scoped.map(0))
    assert torch.equal(scoped[0], scoped[selector])
    assert torch.equal(scoped[selector], scoped[0])
