from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from jaxtyping import Bool, Float
from torch import Tensor

type Reduction = Literal["mean", "sum", "max", "min", "prod"]
type Backend = Literal["auto", "torch", "triton", "torch_eager"]


class ImagePatchFunction[**P](Protocol):
    """Protocol for functions that return image patch shapes given input data."""

    def __call__(
        self, batch_size: int, image_count: int, *args: P.args, **kwargs: P.kwargs
    ) -> list[list[tuple[int, int]]]: ...


class SaliencyQKFunction(Protocol):
    """Protocol for the saliency_qk function used to compute saliency maps from attention scores.

    Args:
        q: Query tensor of shape [B, H, T_gen, D_head]
        k: Key tensor of shape [B, H, T_img, D_head]
        gen_idx: Tensor of shape [B, T_gen] with indices of generated tokens (or -1 for non-generated tokens)
        gen_mask: Bool tensor of shape [B, 1, T_gen] where True indicates valid generated tokens
        img_idx: Tensor of shape [B, T_img] with indices of image tokens (or -1 for non-image tokens)
        img_mask: Bool tensor of shape [B, 1, T_img] where True indicates valid image tokens
        scale: Scaling factor applied to attention scores (e.g., 1/sqrt(D_head))
        layer_reduce: Method to reduce saliency across layers ("mean", "max", etc.)
        head_reduce: Method to reduce saliency across heads ("mean", "max", etc.)
        saliency: Optional existing saliency map to update, of shape [B, T_gen, T_img]
    Returns:
        Saliency map tensor of shape [B, T_gen, T_img]
    """

    def __call__(
        self,
        q: Float[Tensor, "B Hq T_gen D_head"],
        k: Float[Tensor, "B Hkv T_img D_head"],
        gen_idx: Tensor,  # [B, T_gen]
        gen_mask: Bool[Tensor, "B 1 T_gen"],
        img_idx: Tensor,  # [B, T_img]
        img_mask: Bool[Tensor, "B 1 T_img"],
        scale: float,
        saliency: Float[Tensor, "B T_gen T_img"],
    ) -> Float[Tensor, "B T_gen T_img"]: ...


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
        scores: Float[Tensor, "B T_gen T_img"],
        mask: Bool[Tensor, "B T_gen T_img"] | None = None,
    ) -> Float[Tensor, "B T_gen T_img"]: ...


type SelectionSpec = LayerSelect | HeadSelect | Sequence[int] | Mapping[int, Sequence[int]] | None


@dataclass(frozen=True)
class LayerSelect:
    """Configuration for selecting specific layers for saliency accumulation."""

    layers: Sequence[int]
    """Sequence of layer indices to include in saliency accumulation. For example, [0, 2] would select layers 0 and 2."""

    def __iter__(self) -> Iterator[int]:
        return iter(self.layers)


@dataclass(frozen=True)
class HeadSelect:
    """Configuration for selecting specific attention heads for saliency accumulation."""

    heads: Mapping[int, Sequence[int]]
    """Dictionary mapping layer indices to sequences of head indices. For example, {0: [0, 2], 1: [1]} would select heads 0 and 2 from layer 0, and head 1 from layer 1."""

    def __iter__(self) -> Iterator[int]:
        return iter(self.heads)

    def __getitem__(self, layer: int) -> Sequence[int]:
        return self.heads[layer]
