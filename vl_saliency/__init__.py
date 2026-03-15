from transformers import AttentionInterface

from ._hooks.attn import attention_with_saliency
from .context import Saliency

__all__ = ["Saliency"]

AttentionInterface.register("saliency", attention_with_saliency)
