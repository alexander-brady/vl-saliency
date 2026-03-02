from functools import cache

from jaxtyping import Bool, Float, Int
from torch import Tensor

from vl_saliency.backends.reduce import _LAYER_REDUCE
from vl_saliency.types import HeadOp, LayerOp, Reduction, SaliencyQKFunction


@cache
def saliency_qk(
    head_reduce: Reduction,
    layer_reduce: Reduction,
    head_op: HeadOp | None,
    layer_op: LayerOp | None,
) -> SaliencyQKFunction:
    """Creates Triton-optimized saliency QK function with the specified reduction and operation types.

    Args:
        head_reduce: Reduction type for head reduction ('sum' or 'mean'). Other reductions are not currently supported in the Triton backend.
        layer_reduce: Reduction type for layer reduction ('sum', 'mean', 'max', 'min', 'prod').
        head_op: Unused in the Triton backend, must be None.
        layer_op: Optional operation to apply after layer reduction."""
    if head_reduce not in ("sum", "mean") or head_op is not None:
        raise NotImplementedError(
            "Only head_reduce='sum' or 'mean' with no head_op is currently implemented in Triton backend."
        )

    from .autograd import SaliencyQKTriton

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
        out: Float[Tensor, "B T_gen T_img"] = SaliencyQKTriton.apply(
            q,
            k,
            gen_idx,
            gen_mask,
            img_idx,
            img_mask,
            scale,
            saliency,
            head_reduce=head_reduce,
        )  # type: ignore[return-value]

        if layer_op is not None:
            mask = gen_mask[:, :, None] & img_mask[:, None, :]  # [B, T_gen, T_img]
            out = layer_op(out, mask)

        saliency = layer_accum_fn(saliency, out)
        return saliency

    return fn
