from functools import cache

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency.ops.fuse import is_fusable
from vl_saliency.types import HeadOp, LayerOp, Reduction, SaliencyQKFunction

from .reduce import _HEAD_REDUCE, _LAYER_REDUCE
from .scores import _compute_scores


@cache
def saliency_qk_eager(
    head_reduce: Reduction,
    layer_reduce: Reduction,
    head_op: HeadOp | None,
    layer_op: LayerOp | None,
) -> SaliencyQKFunction:
    head_reduce_fn = _HEAD_REDUCE[head_reduce]
    layer_accum_fn = _LAYER_REDUCE[layer_reduce]

    def fn(
        q: Float[Tensor, "B Hq T D"],
        k: Float[Tensor, "B Hkv T D"],
        gen_idx: Int[Tensor, "B T_gen"],
        gen_mask: Bool[Tensor, "B T_gen"],
        img_idx: Int[Tensor, "B T_img"],
        img_mask: Bool[Tensor, "B T_img"],
        scale: float,
        saliency: Float[Tensor, "B T_gen T_img"],
    ) -> Float[Tensor, "B T_gen T_img"]:
        scores = _compute_scores(q, k, gen_idx, img_idx, scale)  # [B, Hq, T_gen, T_img]
        mask = gen_mask[:, None, :, None] & img_mask[:, None, None, :]  # [B, 1, T_gen, T_img]

        if head_op is not None:
            scores = head_op(scores, mask)

        scores = head_reduce_fn(scores, mask)  # [B, T_gen, T_img]

        mask = mask.squeeze(1)  # [B, T_gen, T_img]
        if layer_op is not None:
            scores = layer_op(scores, mask)

        saliency = layer_accum_fn(saliency, scores)
        return saliency

    return fn


@cache
def saliency_qk_compiled(
    head_reduce: Reduction,
    layer_reduce: Reduction,
    head_op: HeadOp | None,
    layer_op: LayerOp | None,
) -> SaliencyQKFunction:
    # For now: if head_op or layer_op is not None, we can't compile because they might not be pure.
    # TODO: Add support for common operations like ReLU, LayerNorm, etc. and compile those.
    full_graph = is_fusable(head_op) and is_fusable(layer_op)
    return torch.compile(
        saliency_qk_eager(head_reduce, layer_reduce, head_op, layer_op),
        mode="max-autotune",
        dynamic=True,  # Variable sequence lengths and masking
        fullgraph=full_graph,
    )
