from transformers import AttentionInterface

from .attn import saliency_attention
from .context import SaliencyContext
from .extractor import SaliencyExtractor

__all__ = ["SaliencyContext", "SaliencyExtractor", "saliency_attention"]

AttentionInterface.register("saliency", saliency_attention)
