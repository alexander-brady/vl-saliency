from collections.abc import Mapping, Sequence

import torch
from jaxtyping import Float, Int
from torch import Tensor

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.core.accum import SaliencyAccumulator
from vl_saliency.types import HeadSelect, LayerSelect


class LayerAccumulator(SaliencyAccumulator):
    """
    Saliency accumulator subclass that only accumulates saliency for a specific subset of layers.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
            subset_select must be a LayerSelect or a sequence of layer indices to include in saliency accumulation.
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        pixel_values (torch.Tensor | None): The input pixel values for the batch, used for dynamic patch shape inference if needed.
        **kwargs: Additional keyword arguments from the forward pass.
    """

    def __init__(
        self,
        config: SaliencyConfig,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        **kwargs,
    ):
        super().__init__(config, input_ids, pixel_values, **kwargs)

        layers = config.subset_select
        if not isinstance(layers, (LayerSelect, Sequence)):
            raise ValueError(
                "Expected subset_select to be a LayerSelect or a sequence of layer indices."
            )

        self.target_layers = set(layers)
        self.current_layer = 0

    def accumulate_qk(self, q: Float[Tensor, "B Hq T D"], k: Float[Tensor, "B Hkv T D"]):
        """Accumulate saliency from the provided query and key tensors if the current layer is in the target layers."""
        if self.current_layer in self.target_layers:
            super().accumulate_qk(q, k)
        self.current_layer += 1


class HeadAccumulator(SaliencyAccumulator):
    """
    Saliency accumulator subclass that only accumulates saliency for a specific subset of attention heads.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
            subset_select must be a HeadSelect or a mapping of layer indices to sequences of head indices to include in saliency accumulation.
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        pixel_values (torch.Tensor | None): The input pixel values for the batch, used for dynamic patch shape inference if needed.
        **kwargs: Additional keyword arguments from the forward pass.
    """

    def __init__(
        self,
        config: SaliencyConfig,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        **kwargs,
    ):
        super().__init__(config, input_ids, pixel_values, **kwargs)

        heads = config.subset_select
        if not isinstance(heads, (HeadSelect, Mapping)):
            raise ValueError(
                "Expected subset_select to be a HeadSelect or a mapping of layer indices to sequences of head indices."
            )

        self.target_heads = {
            layer: torch.tensor(heads[layer], device=input_ids.device) for layer in heads
        }
        self.current_layer = 0

    def accumulate_qk(self, q: Float[Tensor, "B Hq T D"], k: Float[Tensor, "B Hkv T D"]):
        """Accumulate saliency from the provided query and key tensors if the current layer and head are in the target heads."""
        heads = self.target_heads.get(self.current_layer)
        if heads is not None:
            # Check for GQA-style indexing
            Hq = q.shape[1]
            Hkv = k.shape[1]

            q_sel = q.index_select(1, heads)
            if Hq == Hkv:
                k_sel = k.index_select(1, heads)
            else:  # GQA-style indexing where kv heads are indexed separately
                group_size = Hq // Hkv
                kv_heads = heads // group_size
                k_sel = k.index_select(1, kv_heads)

            super().accumulate_qk(q_sel, k_sel)

        self.current_layer += 1
