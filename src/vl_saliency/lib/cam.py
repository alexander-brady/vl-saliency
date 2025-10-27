from .. import transforms as T
from ..core.map import SaliencyMap


def gradcam(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
    """
    Grad-CAM for Vision Transformers.
    Based on https://arxiv.org/abs/1610.02391.
    """
    grad >>= T.relu()
    return grad * attn


def agcam(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
    """
    Attention-Guided Grad-CAM (AGCAM) for Vision-Language Models.
    Based on https://arxiv.org/abs/2402.04563.
    """
    grad >>= T.relu()
    attn >>= T.sigmoid()

    return grad * attn
