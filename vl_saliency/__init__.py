from transformers import AttentionInterface

from .api.config import SaliencyConfig
from .api.out import SaliencyOutput
from .api.saliency import Saliency
from .core.index import Index
from .hooks.attn import attention_with_saliency

__all__ = ["Saliency", "SaliencyConfig", "SaliencyOutput", "Index"]

AttentionInterface.register("saliency", attention_with_saliency)
