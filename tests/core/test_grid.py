import pytest
import torch

from vl_saliency.core.index import Index

from ..utils import ImageSpec

# ------- Test Access Patterns -------


def test_grid_one_batch_one_image(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )

    assert grid._normalize_idx("batch", None) == 0
    assert grid._normalize_idx("image", None) == 0

    assert grid.batch_size == 1
    assert grid.num_images() == grid.num_images(0) == 1
    assert grid.num_tokens() == grid.num_tokens(0) == 3

    assert grid.map(0).shape == (2, 2)
    assert grid.maps_for_image().shape == (3, 2, 2)


def test_grid_one_batch_multiple_images(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1,
        images=[[ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))]],
        gen_tokens=[5],
    )

    assert grid._normalize_idx("batch", None) == 0
    with pytest.raises(IndexError):
        grid._normalize_idx("image", None)

    assert grid.batch_size == 1
    assert grid.num_images() == grid.num_images(0) == 2
    assert grid.num_tokens() == grid.num_tokens(0) == 5

    assert grid.map(0, 0).shape == (2, 2)
    assert grid.map(1, 0).shape == (3, 3)
    assert grid.maps_for_image(1).shape == (5, 3, 3)


def test_grid_multiple_batches_multiple_images(build_sample_grid):
    grid = build_sample_grid(
        batch_size=2,
        images=[
            [ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))],
            [ImageSpec(start=0, size=(1, 1))],
        ],
        gen_tokens=[5, 2],
    )

    with pytest.raises(IndexError):
        grid._normalize_idx("batch", None)
    with pytest.raises(IndexError):
        grid._normalize_idx("image", None)

    assert grid.batch_size == 2

    assert grid.num_images(0) == 2
    assert grid.num_images(1) == 1

    assert grid.num_tokens(0) == 5
    assert grid.num_tokens(1) == 2

    assert grid.map(0, 0, 0).shape == (2, 2)
    assert grid.map(0, 1, 0).shape == (3, 3)

    assert grid.maps_for_image(0, 0).shape == (5, 2, 2)
    assert grid.maps_for_image(1, 0).shape == (2, 1, 1)


# ------- Test Indices -------


def test_grid_accepts_different_index_formats(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1,
        images=[[ImageSpec(start=0, size=(2, 2))]],
        gen_tokens=[5],
    )

    # Using single token index (only valid if batch size is 1 and there's only one image)
    assert torch.equal(grid.map(0), grid[0])

    # Using image and token index (only valid if there's only one batch item)
    assert torch.equal(grid.map(0, 0), grid[0, 0])

    # Using batch, image, and token index
    assert torch.equal(grid.map(0, 0, 0), grid[0, 0, 0])

    # Using Index object
    index = Index(batch_idx=0, image_idx=0, token_idx=0)
    assert torch.equal(grid.map(index), grid[0, 0, 0])

    # Using keyword arguments
    assert torch.equal(grid.map(batch_idx=0, image_idx=0, token_idx=0), grid[0, 0, 0])


def test_grid_invalid_indices(build_sample_grid):
    grid = build_sample_grid(
        batch_size=2,
        images=[
            [ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))],
            [ImageSpec(start=0, size=(1, 1))],
        ],
        gen_tokens=[3, 2],
    )

    with pytest.raises(IndexError):
        grid.map(0, 0, 0, 0)  # Too many indices

    with pytest.raises(IndexError):
        grid._normalize_idx(
            "batch", None
        )  # Can't normalize batch index when there are multiple batches

    with pytest.raises(IndexError):
        grid._normalize_idx(
            "image", None
        )  # Can't normalize image index when there are multiple images

    with pytest.raises(IndexError):
        grid._normalize_idx_input("args", kwargs="kwargs")  # Can't mix args and kwargs

    with pytest.raises(IndexError):
        grid[Index(batch_idx=0, image_idx=0, token_idx=None)]  # None token index is always invalid
