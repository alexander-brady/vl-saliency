from collections.abc import Callable

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from vl_saliency.ops.fuse import fusable
from vl_saliency.types import HeadOp, LayerOp


def make_op(
    fn: Callable[[Float[Tensor, "..."]], Float[Tensor, "..."]],
) -> HeadOp | LayerOp:
    @fusable
    def op(scores: Float[Tensor, "..."], mask: Bool[Tensor, "..."]) -> Float[Tensor, "..."]:
        scores = fn(scores)
        return scores

    op.__name__ = fn.__name__
    op.__doc__ = f"""Applies {fn.__name__} to the saliency scores."""
    return op


relu = make_op(torch.relu)
absolute = make_op(torch.abs)
square = make_op(lambda x: x**2)
sigmoid = make_op(torch.sigmoid)


@make_op
def normalize(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Normalizes saliency scores to [0, 1] range on the last two dimensions."""
    min_val = x.amin(dim=(-2, -1), keepdim=True)
    max_val = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)


@fusable
def softmax(scores: Float[Tensor, "..."], mask: Bool[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Applies masked softmax to the saliency scores on the last two dimensions."""
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    flat = masked_scores.flatten(start_dim=-2)
    return torch.softmax(flat, dim=-1).reshape_as(scores)
