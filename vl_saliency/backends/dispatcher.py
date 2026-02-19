from collections.abc import Callable
from functools import cache

import torch

from vl_saliency._types import Backend
from vl_saliency.backends.torch import compiled_saliency_qk as saliency_qk_torch
from vl_saliency.backends.torch import saliency_qk as saliency_qk_torch_eager
from vl_saliency.backends.triton import saliency_qk as saliency_qk_triton
from vl_saliency.utils.logger import get_logger

logger = get_logger(__name__)


@cache
def get_saliency_qk(backend: Backend, device: torch.device) -> Callable[..., torch.Tensor]:
    """Load the appropriate saliency_qk function based on the specified backend."""

    if backend == "auto":
        if _is_triton_available() and device.type == "cuda":
            backend = "triton"
        elif device.type == "cuda":
            backend = "torch"
        else:
            backend = "torch_eager"
        logger.info(f"Auto-selected backend: {backend}")

    match backend:
        case "torch":
            return saliency_qk_torch()
        case "triton":
            return saliency_qk_triton
        case "torch_eager":
            return saliency_qk_torch_eager
        case _:
            raise ValueError(f"Unsupported backend: {backend}")


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
