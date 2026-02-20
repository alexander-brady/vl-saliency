import pytest
import torch

from vl_saliency.context import SaliencyContext

# ----- Fixtures -----


@pytest.fixture
def simple_inputs():
    """
    Layout:
    [ PAD, IMG, IMG, TXT(before), TXT(gen), TXT(gen) ]
    Only tokens after last IMG are generated.
    """
    pad = 0
    img = 1

    input_ids = torch.tensor([[0, 1, 1, 5, 6, 7]])

    patch_shapes = [[(1, 2)]]  # 2 image tokens

    return input_ids, pad, img, patch_shapes


@pytest.fixture(autouse=True)
def dummy_backend(monkeypatch):
    """
    Replace backend dispatch with a simple deterministic function.
    """

    def fake_get_saliency_qk(*args, **kwargs):
        def _fn(q, k, gen_idx, gen_mask, img_idx, img_mask, scale, saliency):
            # Add 1.0 to all valid positions each call
            return saliency + 1.0

        return _fn

    monkeypatch.setattr(
        "vl_saliency.context.get_saliency_qk",
        fake_get_saliency_qk,
    )
    monkeypatch.setattr(
        "vl_saliency.context.assign_auto",
        lambda device: "fake",
    )


# ----- Tests -----


def test_build_indices(simple_inputs, dummy_backend):
    input_ids, pad, img, patch_shapes = simple_inputs

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    # Two image tokens
    assert ctx.T_img == 2

    # Generated tokens: positions 3, 4, 5
    assert ctx.T_gen == 3
    assert ctx.gen_mask[0].tolist() == [True, True, True]
    assert ctx.img_mask[0].tolist() == [True, True]


def test_reset_creates_zero_saliency(simple_inputs):
    input_ids, pad, img, patch_shapes = simple_inputs

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    assert torch.all(ctx.saliency == 0)
    assert ctx.updates == 0


def test_qk_step_accumulates(simple_inputs):
    input_ids, pad, img, patch_shapes = simple_inputs

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    q = torch.zeros((1, 1, 6, 4))
    k = torch.zeros((1, 1, 6, 4))

    ctx.qk_step(q, k)
    assert ctx.updates == 1
    assert torch.all(ctx.saliency == 1.0)

    ctx.qk_step(q, k)
    assert ctx.updates == 2
    assert torch.all(ctx.saliency == 2.0)


def test_map_returns_correct_shape(simple_inputs):
    input_ids, pad, img, patch_shapes = simple_inputs

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
        layer_reduce="mean",
    )

    q = torch.zeros((1, 1, 6, 4))
    k = torch.zeros((1, 1, 6, 4))
    ctx.qk_step(q, k)

    out = ctx.map(token=0, batch_idx=0, img_idx=0)

    # patch_shapes = (1, 2)
    assert out.shape == (1, 2)


def test_map_validates_indices(simple_inputs):
    input_ids, pad, img, patch_shapes = simple_inputs

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    with pytest.raises(ValueError):
        ctx.map(token=99)

    with pytest.raises(ValueError):
        ctx.map(token=0, batch_idx=99)

    with pytest.raises(ValueError):
        ctx.map(token=0, img_idx=99)
