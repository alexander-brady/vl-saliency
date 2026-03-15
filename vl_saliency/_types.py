from collections.abc import Hashable
from typing import Literal, Protocol

from jaxtyping import Bool, Float
from torch import Tensor

type Backend = Literal["auto", "torch", "triton", "torch_eager"]
"""Backend implemtentations for saliency computation"""

type Reduction = Literal["mean", "sum", "max", "min", "prod", "stack"]
"""Attention score reductions across layers or heads."""


class PatchLayoutFn[**P](Protocol):
    """Functions that compute image patch shapes for given input data."""

    def __call__(
        self, batch_size: int, image_count: int, *args: P.args, **kwargs: P.kwargs
    ) -> list[list[tuple[int, int]]]: ...


class HeadOp(Hashable, Protocol):
    """Operation to be applied to each head's saliency before aggregation. Must be pure.

    Args:
        scores: [B, H, T_gen, T_img]
        mask: [B, 1, T_gen, T_img]

    Returns:
        [B, H, T_gen, T_img]
    """

    def __call__(
        self,
        scores: Float[Tensor, "B H T_gen T_img"],
        mask: Bool[Tensor, "B 1 T_gen T_img"] | None = None,
    ) -> Float[Tensor, "B H T_gen T_img"]: ...


class LayerOp(Hashable, Protocol):
    """Operation to be applied to each layer's saliency before aggregation. Must be pure.

    Args:
        scores: [B, T_gen, T_img]
        mask: [B, T_gen, T_img]

    Returns:
        [B, T_gen, T_img]
    """

    def __call__(
        self,
        scores: Float[Tensor, "B *H T_gen T_img"],
        mask: Bool[Tensor, "B *H T_gen T_img"] | None = None,
    ) -> Float[Tensor, "B *H T_gen T_img"]: ...
