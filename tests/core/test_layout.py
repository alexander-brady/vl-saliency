import pytest
import torch

from vl_saliency.core.layout import SequenceLayout

# -------_build_masks tests -----


def test_layout_build_masks():
    # B=1, S=6
    # img at pos 1 and 3, pad at 5
    input_ids = torch.tensor([[2, 1, 3, 1, 4, 0]])
    pad_id = 0
    img_id = 1

    is_img, is_gen = SequenceLayout._build_masks(input_ids, pad_id, img_id)

    # image tokens
    assert torch.equal(is_img, torch.tensor([[False, True, False, True, False, False]]))

    # last image at pos 3 → generated tokens strictly after 3 and not pad/img
    # pos 4 is gen, pos 5 is pad
    assert torch.equal(is_gen, torch.tensor([[False, False, False, False, True, False]]))


def test_layout_build_masks_no_image_tokens():
    input_ids = torch.tensor([[2, 3, 4, 0]])
    pad_id = 0
    img_id = 1

    is_img, is_gen = SequenceLayout._build_masks(input_ids, pad_id, img_id)

    assert not is_img.any()
    # no image → last_img = S → no generated tokens
    assert not is_gen.any()


# ------- _compact_mask_indices tests -----


def test_layout_compact():
    mask = torch.tensor(
        [
            [False, True, False, True],
            [True, False, False, False],
        ]
    )

    idx, compact_mask, T = SequenceLayout._compact_mask_indices(mask)

    # max count = 2
    assert T == 2

    # row 0 → positions [1, 3]
    assert torch.equal(idx[0], torch.tensor([1, 3], dtype=torch.int32))
    # row 1 → positions [0, -1]
    assert torch.equal(idx[1], torch.tensor([0, -1], dtype=torch.int32))

    assert torch.equal(
        compact_mask,
        torch.tensor(
            [
                [True, True],
                [True, False],
            ]
        ),
    )


def test_layout_compact_all_false():
    mask = torch.zeros((2, 4), dtype=torch.bool)

    idx, compact_mask, T = SequenceLayout._compact_mask_indices(mask)

    assert T == 0
    assert idx.shape == (2, 0)
    assert compact_mask.shape == (2, 0)


# -------_patch_shapes tests -----


def test_layout_patch_shapes_no_pixel_values():
    input_ids = torch.ones((2, 3))

    def patch_fn(batch_size, image_count, **kwargs):
        pytest.fail("patch_fn should not be called when no pixel values are provided")

    patch_shapes = SequenceLayout._patch_shapes(
        image_patch_fn=patch_fn,
        input_ids=input_ids,
        pixel_values=None,
    )

    assert patch_shapes == [[], []]


def test_layout_patch_shapes_valid_call():
    input_ids = torch.ones((2, 3))
    pixel_values = torch.randn((2, 3, 8, 8))

    def patch_fn(batch_size, image_count, **kwargs):
        assert batch_size == 2
        assert image_count == 2
        return [[(2, 2)], [(1, 4)]]

    patch_shapes = SequenceLayout._patch_shapes(
        image_patch_fn=patch_fn,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert patch_shapes == [[(2, 2)], [(1, 4)]]


def test_layout_patch_shapes_mismatched_batch_raises():
    input_ids = torch.ones((2, 3))
    pixel_values = torch.randn((1, 3, 8, 8))  # mismatch

    def bad_patch_fn(batch_size, image_count, **kwargs):
        return [[]]  # only one entry instead of 2

    with pytest.raises(ValueError, match="Number of images"):
        SequenceLayout._patch_shapes(
            image_patch_fn=bad_patch_fn,
            input_ids=input_ids,
            pixel_values=pixel_values,
        )


def test_layout_patch_shapes_wrong_return_length_raises():
    input_ids = torch.ones((2, 3))
    pixel_values = torch.randn((2, 3, 8, 8))

    def bad_patch_fn(batch_size, image_count, **kwargs):
        return [[(2, 2)]]  # only one entry instead of 2

    with pytest.raises(ValueError, match="must return empty patch shapes"):
        SequenceLayout._patch_shapes(
            image_patch_fn=bad_patch_fn,
            input_ids=input_ids,
            pixel_values=pixel_values,
        )


# ------ _image_offsets tests -----


def test_layout_image_offsets_multiple_images():
    patch_shapes = [
        [(2, 2), (1, 3)],  # 4 + 3
        [],  # no images
    ]

    offsets = SequenceLayout._image_offsets(patch_shapes)

    assert offsets == [
        [0, 4, 7],  # 0 → 4 → 7
        [0],
    ]


# -------End-to-end test -----


def test_token_layout_end_to_end(build_config):
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],  # img at 0
            [2, 3, 1, 4],  # img at 2
        ]
    )
    pixel_values = torch.randn((2, 3, 4, 4))

    def patch_fn(batch_size, image_count, **kwargs):
        return [[(2, 2)], [(1, 2)]]

    config = build_config(image_patch_fn=patch_fn)

    layout = SequenceLayout(
        config=config,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    # basic shape metadata
    assert layout.B == 2
    assert layout.S == 4

    # image tokens
    assert layout.T_img == 1
    assert layout.img_mask.shape[0] == 2

    # offsets consistent with patch_fn
    assert layout.image_token_offsets == [
        [0, 4],
        [0, 2],
    ]
