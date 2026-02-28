from functools import wraps

from vl_saliency.types import HeadOp, LayerOp


def fusable[T: HeadOp | LayerOp](fn: T) -> T:
    """Mark a function as fusable, indicating it can be fused into the torch.compile graph."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_fusable = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def is_fusable(fn: HeadOp | LayerOp | None) -> bool:
    """Check if a function is marked as fusable."""
    return fn is None or getattr(fn, "_is_fusable", False)


class FusableMixin:
    """Mixin class to indicate that an operation is fusable."""

    _is_fusable = True
