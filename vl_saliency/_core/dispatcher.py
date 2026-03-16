from functools import cache

import torch

from vl_saliency._backend.interface import SaliencyQKFunction
from vl_saliency._backend.torch import saliency_qk_compiled, saliency_qk_eager
from vl_saliency._backend.triton import saliency_qk as saliency_qk_triton
from vl_saliency._logging import get_logger
from vl_saliency._types import Backend, HeadOp, LayerOp, Reduction

logger = get_logger(__name__)


def assign_auto(device: torch.device, head_reduce: Reduction) -> Backend:
    """Auto select saliency backend based on device and reduction spec."""
    if _is_triton_available() and device.type == "cuda" and head_reduce in ("sum", "mean"):
        return "triton"
    elif device.type == "cuda":
        return "torch"
    else:
        return "torch_eager"


@cache
def get_qk_accumulator(
    backend: Backend,
    head_reduce: Reduction,
    layer_reduce: Reduction,
    head_op: HeadOp | None,
    layer_op: LayerOp | None,
) -> SaliencyQKFunction:
    """Load the appropriate saliency_qk function based on the specified backend and configuration."""
    match backend:
        case "torch":
            try:
                return saliency_qk_compiled(head_reduce, layer_reduce, head_op, layer_op)
            except Exception:
                logger.warning_once(
                    "Failed to compile `saliency` kernel, falling back to eager mode."
                )
                return saliency_qk_eager(head_reduce, layer_reduce, head_op, layer_op)
        case "triton":
            return saliency_qk_triton(head_reduce, layer_reduce, head_op, layer_op)
        case "torch_eager":
            return saliency_qk_eager(head_reduce, layer_reduce, head_op, layer_op)
        case "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            selected_backend = assign_auto(device, head_reduce=head_reduce)
            logger.info_once(f"Auto-selected backend: {selected_backend} (device: {device})")
            return get_qk_accumulator(
                selected_backend, head_reduce, layer_reduce, head_op, layer_op
            )
        case _:
            raise ValueError(
                f"Invalid backend: {backend}. Must be one of `auto`, `triton`, `torch`, `torch_eager`."
            )


@cache
def _is_triton_available() -> bool:
    try:
        import triton
        import triton.language  # type: ignore[import]  # noqa: F401
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        _ = torch.cuda.get_device_capability()
    except Exception:
        return False

    return True
