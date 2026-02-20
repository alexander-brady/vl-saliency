from vl_saliency._types import HeadOp, LayerOp, Reduction, SaliencyQKFunction


def saliency_qk(
    head_reduce: Reduction,
    layer_reduce: Reduction,
    head_op: HeadOp | None,
    layer_op: LayerOp | None,
) -> SaliencyQKFunction:
    raise NotImplementedError("Triton backend is not yet implemented")
