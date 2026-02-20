from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from vl_saliency.context import SaliencyContext


def saliency_attention(
    module: torch.nn.Module,
    query: Float[Tensor, "B Hq T D_head"],
    key: Float[Tensor, "B Hkv T D_head"],
    value: Float[Tensor, "B Hkv T D_head"],
    attention_mask: Float[Tensor, "B 1 T T"] | None,
    saliency: SaliencyContext,
    **kwargs,
) -> tuple[Float[Tensor, "B Hq T D_head"], Float[Tensor, "B Hq T T"] | None]:
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
    attention_interface: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = (
        ALL_ATTENTION_FUNCTIONS.get_interface(saliency.attn_implementation, sdpa_attention_forward)
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

    # Compute saliency map
    saliency.qk_step(query, key)

    # Return the attention output and weights for further processing
    return attn_output, attn_weights
