from dataclasses import dataclass

import pytest
import torch

from vl_saliency.core.grid import SaliencyGrid
from vl_saliency.core.index import Index
from vl_saliency.core.layout import SequenceLayout

# ------- Helper classes and fixtures for testing -------


@dataclass
class ImageSpec:
    start: int
    size: tuple[int, int]


class DummyLayout(SequenceLayout):
    def __init__(
        self,
        B: int,
        patch_shapes: list[list[tuple[int, int]]],
        image_token_offsets: list[list[int]],
        gen_mask: torch.Tensor,
    ):
        self.B = B
        self.patch_shapes = patch_shapes
        self.image_token_offsets = image_token_offsets
        self.gen_mask = gen_mask


@pytest.fixture
def build_sample_grid():
    """
    Builds SaliencyGrids as follows:
    Batch 0:
        Image 0: starts at 0, size (2, 2) → occupies indices [0, 4)
        Image 1: starts at 4, size (3, 3) → occupies indices [4, 13)
        Gen tokens: 5 → token indices [0, 5)
    Batch 1:
        Image 0: starts at 0, size (1, 1) → occupies indices [0, 1)
        Gen tokens: 2 → token indices [0, 2)
    Tensor:
        Shape: (B=2, T_gen=5, T_img=13) → accommodates all tokens and image patches
        Values: Sequential integers for easy verification
    """

    def _build(
        batch_size: int, images: list[list[ImageSpec]], gen_tokens: list[int]
    ) -> SaliencyGrid:
        max_gen_tokens = max(gen_tokens)

        lowest_start = min((spec.start for batch in images for spec in batch), default=0)
        highest_end = max(
            (spec.start + spec.size[0] * spec.size[1] for batch in images for spec in batch),
            default=0,
        )

        tensor_shape = (batch_size, max_gen_tokens, highest_end - lowest_start)
        tensor = torch.arange(
            tensor_shape[0] * tensor_shape[1] * tensor_shape[2],
            dtype=torch.float32,
        ).reshape(tensor_shape)

        layout = DummyLayout(
            B=batch_size,
            patch_shapes=[[spec.size for spec in batch] for batch in images],
            image_token_offsets=[[spec.start for spec in batch] for batch in images],
            gen_mask=torch.tensor(
                [
                    [1] * num_tokens + [0] * (max_gen_tokens - num_tokens)
                    for num_tokens in gen_tokens
                ]
            ),
        )
        return SaliencyGrid(tensor=tensor, layout=layout)

    return _build


# ------- Test Access Patterns -------


def test_one_batch_one_image(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1, images=[[ImageSpec(start=0, size=(2, 2))]], gen_tokens=[3]
    )
    assert grid.batch_size == 1
    assert grid.num_images() == grid.num_images(0) == 1
    assert grid.num_tokens() == grid.num_tokens(0) == 3

    assert grid.map(0).shape == (2, 2)
    assert grid.image_maps(0).shape == (3, 2, 2)

    assert grid._validate_batch_idx(None) == 0
    assert grid._validate_img_idx(None, 0) == 0


def test_one_batch_multiple_images(build_sample_grid):
    grid = build_sample_grid(
        batch_size=1,
        images=[[ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))]],
        gen_tokens=[5],
    )

    assert grid.batch_size == 1
    assert grid.num_images() == grid.num_images(0) == 2
    assert grid.num_tokens() == grid.num_tokens(0) == 5

    assert grid.map(0, 0).shape == (2, 2)
    assert grid.map(1, 0).shape == (3, 3)


def test_multiple_batches_multiple_images(build_sample_grid):
    grid = build_sample_grid(
        batch_size=2,
        images=[
            [ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))],
            [ImageSpec(start=0, size=(1, 1))],
        ],
        gen_tokens=[5, 2],
    )

    assert grid.batch_size == 2

    assert grid.num_images(0) == 2
    assert grid.num_images(1) == 1

    assert grid.num_tokens(0) == 5
    assert grid.num_tokens(1) == 2

    assert grid.map(0, 0, 0).shape == (2, 2)
    assert grid.map(0, 1, 0).shape == (3, 3)


# ------- Test Indices -------


def test_accepts_different_index_formats(build_sample_grid):
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
    index = Index(batch_idx=0, img_idx=0, token_idx=0)
    assert torch.equal(grid.map(index), grid[0, 0, 0])


def test_invalid_indices(build_sample_grid):
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
        grid._validate_batch_idx(None)  # Ambiguous batch index with multiple batches

    with pytest.raises(IndexError):
        grid._validate_batch_idx(2)  # Invalid batch index

    with pytest.raises(IndexError):
        grid._validate_img_idx(0, None)  # Ambiguous image index with multiple images in batch

    with pytest.raises(IndexError):
        grid._validate_img_idx(0, 2)  # Invalid image index for batch 0

    with pytest.raises(IndexError):
        grid._validate_token_idx(0, -1)  # Invalid token index for batch 0

    with pytest.raises(IndexError):
        grid._validate_token_idx(0, 5)  # Invalid token index for batch 0 (only 3 tokens)

    with pytest.raises(IndexError):
        index = Index(batch_idx=0, img_idx=0, token_idx=None)
        grid._validate_index(index)  # Token index must be specified
