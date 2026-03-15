from collections.abc import Callable
from types import MappingProxyType

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from vl_saliency._types import Reduction


def _reduce_mean(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B T_gen T_img"]:
    masked_scores = scores * mask
    return masked_scores.sum(dim=1) / scores.shape[1]  # [B, T_gen, T_img]


def _reduce_sum(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B T_gen T_img"]:
    masked_scores = scores * mask
    return masked_scores.sum(dim=1)  # [B, T_gen, T_img]


def _reduce_max(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B T_gen T_img"]:
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return masked_scores.amax(dim=1)  # [B, T_gen, T_img]


def _reduce_min(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B T_gen T_img"]:
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).max)
    return masked_scores.amin(dim=1)  # [B, T_gen, T_img]


def _reduce_prod(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B T_gen T_img"]:
    one = torch.ones((), dtype=scores.dtype, device=scores.device)
    masked_scores = scores.masked_fill(~mask, one)
    return masked_scores.prod(dim=1)  # [B, T_gen, T_img]


def _stack_head(
    scores: Float[Tensor, "B H T_gen T_img"], mask: Bool[Tensor, "B 1 T_gen T_img"]
) -> Float[Tensor, "B H T_gen T_img"]:
    masked_scores = scores * mask
    return masked_scores  # [B, H, T_gen, T_img]


def _stack_layer(
    base: Float[Tensor, "B L ... T_gen T_img"], new: Float[Tensor, "B L T_gen T_img"]
) -> Float[Tensor, "B L+1 ... T_gen T_img"]:
    return torch.cat([base, new.unsqueeze(1)], dim=1)  # [B, L+1, ... T_gen T_img]


_HEAD_REDUCE: MappingProxyType[
    Reduction,
    Callable[
        [Float[Tensor, "B H T_gen T_img"], Bool[Tensor, "B 1 T_gen T_img"]],
        Float[Tensor, "B ... T_gen T_img"],
    ],
] = MappingProxyType(
    {
        "mean": _reduce_mean,
        "sum": _reduce_sum,
        "max": _reduce_max,
        "min": _reduce_min,
        "prod": _reduce_prod,
        "stack": _stack_head,
    }
)

_LAYER_REDUCE: MappingProxyType[
    Reduction,
    Callable[
        [Float[Tensor, "B ... T_gen T_img"], Float[Tensor, "B ... L T_gen T_img"]],
        Float[Tensor, "B ... T_gen T_img"],
    ],
] = MappingProxyType(
    {
        "mean": torch.add,  # We'll divide by count in the end, so just sum here
        "sum": torch.add,
        "max": torch.maximum,
        "min": torch.minimum,
        "prod": torch.mul,
        "stack": _stack_layer,
    }
)

__all__ = ["_HEAD_REDUCE", "_LAYER_REDUCE"]
