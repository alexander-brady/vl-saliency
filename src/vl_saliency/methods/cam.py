import torch
import torch.nn.functional as F

from .registry import register


@register("gradcam")
def gradcam(attn: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Grad-CAM for Vision Transformers.
    Based on https://arxiv.org/abs/1610.02391.

    Args:
        attn (torch.Tensor): Attention weights of shape [num_layers, num_heads, seq_len].
        grad (torch.Tensor): Gradients of shape [num_layers, num_heads, seq_len].

    Returns:
        torch.Tensor: Saliency map of shape [num_layers, num_heads, seq_len].
    """
    grad = F.relu(grad)
    return grad * attn


@register("agcam")
def agcam(attn: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Attention-Guided Grad-CAM (AGCAM) for Vision-Language Models.
    Based on https://arxiv.org/abs/2402.04563.

    Args:
        attn (torch.Tensor): Attention weights of shape [num_layers, num_heads, seq_len].
        grad (torch.Tensor): Gradients of shape [num_layers, num_heads, seq_len].

    Returns:
        torch.Tensor: Saliency map of shape [num_layers, num_heads, seq_len].
    """
    grad = F.relu(grad)
    attn = torch.sigmoid(attn)
    return grad * attn
