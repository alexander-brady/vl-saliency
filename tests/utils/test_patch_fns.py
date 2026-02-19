import pytest
import torch

from vl_saliency.utils.patch_fns import StaticPatches, image_thw_to_patches

# ------ Test StaticPatches ------


def test_static_patches_single_image():
    sp = StaticPatches(height=16, width=32)
    result = sp(batch_size=1, image_count=1)

    assert result == [[(16, 32)]]


def test_static_patches_zero_images():
    sp = StaticPatches(height=4, width=4)
    result = sp(batch_size=2, image_count=0)

    assert result == [
        [(4, 4)],
        [(4, 4)],
    ]  # Still returns patch shape for each batch item, but with empty image list


# ------ Test image_thw_to_patches ------


def test_image_thw_to_patches_valid_input():
    # T, H, W
    image_grid_thw = torch.tensor(
        [
            [1, 32, 64],
            [1, 16, 48],
        ]
    )

    result = image_thw_to_patches(
        batch_size=2,
        image_count=2,
        image_grid_thw=image_grid_thw,
    )

    # H//2, W//2
    expected = [
        [(16, 32)],
        [(8, 24)],
    ]

    assert result == expected


def test_image_thw_to_patches_mismatch_count():
    image_grid_thw = torch.tensor(
        [
            [1, 32, 64],
        ]
    )

    with pytest.raises(ValueError, match="does not match number of images"):
        image_thw_to_patches(
            batch_size=2,
            image_count=2,
            image_grid_thw=image_grid_thw,
        )


def test_image_thw_to_patches_integer_division():
    # Odd H/W should floor via integer division
    image_grid_thw = torch.tensor(
        [
            [1, 33, 65],
        ]
    )

    result = image_thw_to_patches(
        batch_size=1,
        image_count=1,
        image_grid_thw=image_grid_thw,
    )

    assert result == [[(16, 32)]]  # 33//2=16, 65//2=32
