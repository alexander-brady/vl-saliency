from transformers import AttentionInterface

from .attn import saliency_attention
from .saliency import Saliency

__all__ = ["Saliency"]

AttentionInterface.register("saliency", saliency_attention)
