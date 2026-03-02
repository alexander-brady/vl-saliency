import matplotlib
import numpy as np
import pytest
import torch
from PIL import Image

from vl_saliency.core.scoped import ScopedSaliencyGrid

"""Dummy Saliency Grid as follows:
Batch 0: (not used here)
    Image 0: starts at 0, size (2, 2) → occupies indices [0, 4)
    Image 1: starts at 4, size (3, 3) → occupies indices [4, 13)
    Gen tokens: 5 → token indices [13, 18)
    Gen Mask: [1, 1, 1, 1, 1]
Batch 1: 
    Image 0: starts at 0, size (1, 1) → occupies indices [0, 1)
    Gen tokens: 2 → token indices [1, 3)
    Gen Mask: [1, 1, 0, 0, 0]
Tensor:
    Shape: (B=2, T_gen=5, T_img=13) → accommodates all tokens and image patches
    Values: Sequential integers for easy verification
"""

# ------- Input IDs for the above grid: -------


def dummy_input_ids():
    batch_0 = [0] * 13 + [10, 11, 12, 13, 14]  # 13 image tokens + 5 gen tokens
    batch_1 = [0] * 1 + [20, 21] + [-1] * 15  # 1 image token + 2 gen tokens + padding
    return torch.tensor([batch_0, batch_1])

# ------- Scoped Saliency Grid Tests -------


def test_scoped_init(dummy_saliency_grid):
    scoped = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0)

    assert scoped.batch_idx == 1
    assert scoped.image_idx == 0
    assert scoped.num_tokens == 2
    assert scoped.gen_start_idx == 1  # after the single image token at index 0
    assert scoped.gen_end_idx == 3  # 2 gen tokens after start idx

    with pytest.raises(ValueError):
        _ = scoped.gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_input_tokens

    scoped_2 = dummy_saliency_grid.scope(batch_idx=1, image_idx=0)
    assert scoped_2.batch_idx == 1
    assert scoped_2.image_idx == 0
    assert scoped_2.num_tokens == 2
    assert scoped_2.gen_start_idx == 1
    assert scoped_2.gen_end_idx == 3

    assert torch.equal(scoped.maps, scoped_2.maps)
    assert torch.equal(scoped.map(0), scoped_2.map(0))


def test_scoped_input_ids(dummy_saliency_grid):
    input_ids = dummy_input_ids()
    scoped = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0, input_ids=input_ids)

    assert scoped.input_ids is not None
    assert scoped.input_ids.shape == (18,)  # Total tokens for batch 1
    assert torch.equal(scoped.input_ids, input_ids[1])  # Scoped to batch_idx=1

    assert torch.equal(scoped.gen_tokens, input_ids[1, 1:3])

    with pytest.raises(ValueError):
        _ = scoped.decoded_gen_tokens
    with pytest.raises(ValueError):
        _ = scoped.decoded_input_tokens

    scoped_2 = dummy_saliency_grid.scope(batch_idx=1, image_idx=0, input_ids=input_ids[1])
    assert torch.equal(scoped_2.input_ids, scoped.input_ids)
    assert torch.equal(scoped_2.gen_tokens, scoped.gen_tokens)


def test_scoped_processor_tokenizer(dummy_saliency_grid, dummy_tokenizer, dummy_processor):
    input_ids = dummy_input_ids()
    scoped = ScopedSaliencyGrid(
        dummy_saliency_grid,
        batch_idx=1,
        image_idx=0,
        input_ids=input_ids,
        processor=dummy_tokenizer,
    )

    print(dummy_tokenizer)
    print(scoped._tok)

    assert scoped._tok is not None

    assert torch.equal(scoped.gen_tokens, input_ids[1, 1:3])  # batch 1, gen tokens
    assert scoped.decoded_gen_tokens == ["token_20", "token_21"]
    assert scoped.decoded_input_tokens == ["token_0"] + ["token_20", "token_21"] + ["token_-1"] * 15

    scoped_2 = dummy_saliency_grid.scope(
        batch_idx=1, image_idx=0, input_ids=input_ids[1], processor=dummy_processor
    )
    assert scoped_2._tok is not None
    assert torch.equal(scoped_2.gen_tokens, scoped.gen_tokens)
    assert scoped_2.decoded_gen_tokens == scoped.decoded_gen_tokens
    assert scoped_2.decoded_input_tokens == scoped.decoded_input_tokens


def test_scoped_selector(dummy_saliency_grid):
    input_ids = dummy_input_ids()
    scoped = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0, input_ids=input_ids)

    def selector(scoped):
        return 0

    assert torch.equal(scoped.map(selector), scoped.map(0))
    assert torch.equal(scoped[0], scoped[selector])
    assert torch.equal(scoped[selector], scoped[0])


# ------- Visualization Tests -------

matplotlib.use("Agg")


def test_scoped_plot(dummy_saliency_grid):
    image = Image.new("RGB", (16, 16), color="white")
    scoped = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0, image=image)
    fig = scoped.plot(0, image=image, cmap="viridis", alpha=0.5)

    from vl_saliency.viz.overlay import plot

    expected_fig = plot(scoped.map(0), image=image, cmap="viridis", alpha=0.5)

    def fig_to_array(fig):
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    fig_array = fig_to_array(fig)
    expected_array = fig_to_array(expected_fig)
    assert np.allclose(fig_array, expected_array)


def test_scoped_visualize(dummy_saliency_grid, dummy_tokenizer):
    scoped = ScopedSaliencyGrid(dummy_saliency_grid, batch_idx=1, image_idx=0)

    with pytest.raises(ValueError):
        _ = scoped.visualize_tokens()  # no input_ids or processor/tokenizer provided
    scoped.input_ids = dummy_input_ids()[1]  # batch 1 input ids

    scoped._tok = dummy_tokenizer
    out = scoped.visualize_tokens(return_html=True)

    from vl_saliency.viz.tokens import render_token_ids

    expected_out = render_token_ids(
        token_ids=dummy_input_ids()[1].tolist(),
        processor=dummy_tokenizer,
        return_html=True,
        skip_tokens=(0, -1),  # skip image token and padding
        gen_start=1,
        only_number_generated=True,
    )

    assert out == expected_out
