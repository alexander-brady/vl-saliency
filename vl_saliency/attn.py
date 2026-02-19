from collections.abc import Callable

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vl_saliency.backends import get_saliency_qk
from vl_saliency.context import SaliencyContext


def saliency_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    saliency: SaliencyContext,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute attention output and weights,
    while also computing and updating the saliency map based on the provided context.

    Args:
        module (torch.nn.Module): The attention module being called, used to determine the attention implementation for saliency extraction.
        query (torch.Tensor): The query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (torch.Tensor): The key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        value (torch.Tensor): The value tensor of shape (batch_size, num_heads, seq_len_v, head_dim).
        attention_mask (torch.Tensor | None): The attention mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k) or None.
        saliency (SaliencyContext): The saliency context object used to compute and update the saliency map.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: A tuple containing the attention output tensor and the attention weights tensor (or None if not returned by the attention implementation).
    """
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        saliency.attn_implementation, sdpa_attention_forward
    )

    # Standard attention forward pass to get output and weights
    attn_output, attn_weights = attention_interface(
        module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )

    # Compute saliency map using the appropriate backend
    saliency_qk = get_saliency_qk(saliency.backend, query.device)
    sm = saliency_qk(
        q=query,
        k=key,
        gen_idx=saliency.gen_token_idx,
        gen_mask=saliency.gen_mask,
        img_idx=saliency.img_token_idx,
        img_mask=saliency.img_mask,
        scale=saliency.scale,
        layer_reduce=saliency.layer_reduce,
        head_reduce=saliency.head_reduce,
        saliency=saliency.saliency,
    )

    # Update the saliency map in the context with the computed values
    saliency.update(sm)

    # Return the attention output and weights for further processing
    return attn_output, attn_weights
