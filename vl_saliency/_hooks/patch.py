from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType

from transformers import PreTrainedModel

from vl_saliency._hooks.forward import build_saliency_forward
from vl_saliency.config import SaliencyConfig

_PATCH_ATTR_NAME = "_vl_saliency_patch"


@dataclass(frozen=True)
class PatchState:
    """Frame to store the original state of the model's forward method
    and attention implementation before patching, allowing for restoration later."""

    prev_forward: Callable | None = None
    prev_attn_impl: str | None = None


def is_patched(model: PreTrainedModel) -> bool:
    """Checks if the model has already been patched for saliency computation."""
    return hasattr(model, _PATCH_ATTR_NAME)


def apply_forward_patch(model: PreTrainedModel, config: SaliencyConfig):
    """Applies the forward patch to the model for saliency computation."""

    if is_patched(model):
        return  # Already enabled

    original_forward = model.forward
    original_attn_impl = model.config.text_config._attn_implementation

    new_forward = build_saliency_forward(
        config=config,
        attn_implementation=original_attn_impl,
        forward=original_forward,
    )

    patch = PatchState(prev_forward=original_forward, prev_attn_impl=original_attn_impl)

    setattr(model, _PATCH_ATTR_NAME, patch)

    model.set_attn_implementation({"text_config": "saliency"})
    model.forward = MethodType(new_forward, model)


def restore_forward(model: PreTrainedModel):
    """Restores the model's original forward method and attention implementation."""
    if not is_patched(model):
        return  # Nothing to restore

    patch = getattr(model, _PATCH_ATTR_NAME)

    if patch.prev_forward:
        model.forward = patch.prev_forward

    if patch.prev_attn_impl:
        model.set_attn_implementation({"text_config": patch.prev_attn_impl})

    delattr(model, _PATCH_ATTR_NAME)
