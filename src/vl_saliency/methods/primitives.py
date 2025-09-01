import torch
import torch.nn.functional as F

from .registry import register


@register("attn_raw", "attn")
def attn_raw(
    attn: torch.Tensor, grad: torch.Tensor, *, sigmoid: bool = False
) -> torch.Tensor:
    """
    Use the attention map.

    Args:
        attn (torch.Tensor): The attention maps. [num_layers, heads, tokens]
        grad (torch.Tensor): The gradients.      [num_layers, heads, tokens]
        sigmoid (bool): Whether to apply sigmoid.

    Returns:
        torch.Tensor: The attention maps.        [num_layers, heads, tokens]
    """
    if sigmoid:
        attn = torch.sigmoid(attn)

    return attn


@register("grad_raw", "grad")
def grad_raw(
    attn: torch.Tensor, grad: torch.Tensor, *, relu: bool = False, abs: bool = False
) -> torch.Tensor:
    """
    Use the gradients of the attention map.

    Args:
        attn (torch.Tensor): The attention maps. [num_layers, heads, tokens]
        grad (torch.Tensor): The gradients.      [num_layers, heads, tokens]
        relu (bool): Whether to apply ReLU.
        abs (bool): Whether to apply absolute value.

    Returns:
        torch.Tensor: The processed gradients.   [num_layers, heads, tokens]
    """
    if relu:
        grad = F.relu(grad)

    if abs:
        grad = grad.abs()

    return grad
