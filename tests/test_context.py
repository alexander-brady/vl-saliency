import pytest
import torch

from vl_saliency.context import SaliencyContext


def test_init_builds_indices_offsets_and_reset():
    # batch 0: 4 image tokens (2 images: 1x2 and 2x1), then 2 generated tokens
    # batch 1: no image tokens (hits has_img=False branch), so no generated tokens (pos > S never true)
    pad = 0
    img = 9
    input_ids = torch.tensor(
        [
            [pad, img, img, img, img, 11, 12],  # last img at pos 4 => gen at pos 5,6
            [pad, 5, 6, pad, 7, 8, pad],  # no img => no gen
        ],
        dtype=torch.long,
    )

    patch_shapes = [
        [(1, 2), (2, 1)],  # 2 + 2 = 4 image tokens
        [],  # no images for batch 1
    ]

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    assert ctx.device.type == "cpu"
    assert ctx.B == 2
    assert ctx.T_img == 4
    assert ctx.T_gen == 2

    # reset() happened in __init__
    assert ctx.saliency.shape == (2, 2, 4)
    assert torch.all(ctx.saliency == 0)
    assert ctx.updates == 0

    # offsets for batch 0: [0, 2, 4]
    assert ctx.image_token_offsets[0] == [0, 2, 4]
    # batch 1 has no images -> offsets should still exist, starts with [0]
    assert ctx.image_token_offsets[1] == [0]

    # masks/indices sanity
    assert ctx.img_mask[0].tolist() == [True, True, True, True]
    assert ctx.img_mask[1].tolist() == [False, False, False, False]
    assert ctx.gen_mask[0].tolist() == [True, True]
    assert ctx.gen_mask[1].tolist() == [False, False]


def test_update_and_map_mean_divides_by_updates_and_views_shape():
    pad = 0
    img = 9
    input_ids = torch.tensor(
        [[img, img, img, img, 1, 2]], dtype=torch.long
    )  # 2 generated tokens after last img
    patch_shapes = [[(1, 2), (2, 1)]]  # total 4 tokens

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
        layer_reduce="mean",
    )

    # Put distinct values into saliency so we can verify slicing+reshape+division
    sal = torch.arange(ctx.B * ctx.T_gen * ctx.T_img, dtype=torch.float32).view(
        ctx.B, ctx.T_gen, ctx.T_img
    )

    ctx.update(sal)
    ctx.update(sal)  # updates=2, saliency unchanged but should divide by 2 in map()

    # token=0, img_idx=0 -> first image is 1x2, offset 0..1
    m0 = ctx.map(token=0, batch_idx=0, img_idx=0)
    assert m0.shape == (1, 2)
    expected0 = sal[0, 0, 0:2] / 2
    assert torch.allclose(m0.flatten(), expected0)

    # token=1, img_idx=1 -> second image is 2x1, offset 2..3
    m1 = ctx.map(token=1, batch_idx=0, img_idx=1)
    assert m1.shape == (2, 1)
    expected1 = sal[0, 1, 2:4] / 2
    assert torch.allclose(m1.flatten(), expected1)


def test_map_no_division_when_layer_reduce_not_mean():
    pad = 0
    img = 9
    input_ids = torch.tensor(
        [[img, img, 3]], dtype=torch.long
    )  # last img at pos 1 => gen at pos 2 (1 token)
    patch_shapes = [[(1, 2)]]  # 2 image tokens

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
        layer_reduce="prod",  # anything != "mean" avoids division branch
    )

    sal = torch.tensor([[[10.0, 20.0]]])  # (B=1,T_gen=1,T_img=2)
    ctx.update(sal)
    ctx.update(sal)  # updates=2 but should NOT divide

    m = ctx.map(token=0, batch_idx=0, img_idx=0)
    assert m.shape == (1, 2)
    assert torch.allclose(m.flatten(), sal[0, 0])


def test_map_validation_errors():
    pad = 0
    img = 9
    input_ids = torch.tensor(
        [[img, img, 3, 4]], dtype=torch.long
    )  # gen tokens at pos 2,3 => T_gen=2
    patch_shapes = [[(1, 2)]]

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )
    ctx.update(torch.zeros((1, 2, 2)))

    with pytest.raises(ValueError, match="Batch index .* out of bounds"):
        ctx.map(token=0, batch_idx=1, img_idx=0)

    with pytest.raises(ValueError, match="Image index .* out of bounds"):
        ctx.map(token=0, batch_idx=0, img_idx=1)

    # token out of bounds (>= T_gen)
    with pytest.raises(ValueError, match="Generated token index .* out of bounds"):
        ctx.map(token=2, batch_idx=0, img_idx=0)

    # token in range but not generated: construct a context where T_gen>0 globally,
    # but a particular batch has no generated tokens (gen_mask False)
    input_ids2 = torch.tensor(
        [
            [img, img, 3, 4],  # has gen
            [img, img, pad, pad],  # no gen
        ],
        dtype=torch.long,
    )
    patch_shapes2 = [[(1, 2)], [(1, 2)]]
    ctx2 = SaliencyContext(
        input_ids=input_ids2,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes2,
        scale=1.0,
    )
    ctx2.update(torch.zeros((2, ctx2.T_gen, ctx2.T_img)))

    assert ctx2.T_gen == 2
    assert ctx2.gen_mask[1, 0].item() is False
    with pytest.raises(ValueError, match="Generated token index .* out of bounds"):
        ctx2.map(token=0, batch_idx=1, img_idx=0)


def test_build_indices_all_empty_hits_default_zeros_and_reset_shapes():
    pad = 0
    img = 9
    # all pad => no img tokens; also no gen tokens
    input_ids = torch.tensor([[pad, pad], [pad, pad]], dtype=torch.long)
    patch_shapes = [[], []]

    ctx = SaliencyContext(
        input_ids=input_ids,
        pad_token_id=pad,
        image_token_id=img,
        patch_shapes=patch_shapes,
        scale=1.0,
    )

    assert ctx.T_img == 0
    assert ctx.T_gen == 0
    assert ctx.img_token_idx.shape == (2, 0)
    assert ctx.gen_token_idx.shape == (2, 0)
    assert ctx.img_mask.shape == (2, 0)
    assert ctx.gen_mask.shape == (2, 0)

    # reset created an empty saliency tensor with correct shape
    assert ctx.saliency.shape == (2, 0, 0)
    assert ctx.updates == 0
