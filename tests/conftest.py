from typing import Any

import pytest
import torch
from transformers import PreTrainedConfig

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.core.grid import SaliencyGrid
from vl_saliency.core.layout import SequenceLayout
from vl_saliency.utils.patch_fns import FixedPatchLayout

from .utils import ImageSpec

# ------- Configuration -----


@pytest.fixture
def build_config():
    def _mk(**overrides) -> SaliencyConfig:
        base = dict[str, Any](
            pad_token_id=0,
            image_token_id=1,
            image_patch_fn=FixedPatchLayout(16, 16),
            layer_reduce="mean",
            layer_op=None,
            head_reduce="mean",
            head_op=None,
            backend="auto",
            attn_scale=0.25,
        )
        base.update(overrides)
        return SaliencyConfig(**base)

    return _mk


# ------- Processing -------


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


# -------Models -------


class DummyModelConfig(PreTrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class DummyModel:
    def __init__(self, config):
        self.config = config


@pytest.fixture
def build_model_config():
    def _mk(**overrides) -> DummyModelConfig:
        return DummyModelConfig(**overrides)

    return _mk


@pytest.fixture
def build_model(build_model_config):
    def _mk(**config_overrides) -> DummyModel:
        config = build_model_config(**config_overrides)
        return DummyModel(config)

    return _mk


# -------Saliency grid -----


class DummyLayout(SequenceLayout):
    def __init__(
        self,
        B: int,
        patch_shapes: list[list[tuple[int, int]]],
        image_token_offsets: list[list[int]],
        gen_mask: torch.Tensor,
        gen_token_idx: torch.Tensor,
    ):
        self.B = B
        self.patch_shapes = patch_shapes
        self.image_token_offsets = image_token_offsets
        self.gen_mask = gen_mask
        self.gen_token_idx = gen_token_idx


@pytest.fixture
def build_sample_grid():

    def _build(
        batch_size: int, images: list[list[ImageSpec]], gen_tokens: list[int]
    ) -> SaliencyGrid:
        max_gen_tokens = max(gen_tokens)
        gen_start_indices = [
            max((spec.start + spec.size[0] * spec.size[1] for spec in batch), default=0)
            for batch in images
        ]

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
                    [True] * num_tokens + [False] * (max_gen_tokens - num_tokens)
                    for num_tokens in gen_tokens
                ]
            ),
            gen_token_idx=torch.tensor(
                [
                    list(range(gen_start, gen_start + num_tokens))
                    + [-1] * (max_gen_tokens - num_tokens)
                    for gen_start, num_tokens in zip(gen_start_indices, gen_tokens, strict=True)
                ]
            ),
        )
        return SaliencyGrid(tensor=tensor, layout=layout)

    return _build


@pytest.fixture
def dummy_saliency_grid(build_sample_grid):
    """Builds SaliencyGrids as follows:
    Batch 0:
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
    return build_sample_grid(
        batch_size=2,
        images=[
            [ImageSpec(start=0, size=(2, 2)), ImageSpec(start=4, size=(3, 3))],
            [ImageSpec(start=0, size=(1, 1))],
        ],
        gen_tokens=[5, 2],
    )
