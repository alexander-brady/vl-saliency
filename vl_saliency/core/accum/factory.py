from collections.abc import Mapping, Sequence

from jaxtyping import Float
from torch import Tensor

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.core.accum.base import SaliencyAccumulator
from vl_saliency.core.accum.subset import HeadAccumulator, LayerAccumulator
from vl_saliency.types import HeadSelect, LayerSelect


def build_accumulator(
    config: SaliencyConfig,
    input_ids: Tensor,
    pixel_values: Float[Tensor, "B C H W"] | None = None,
    **kwargs,
) -> SaliencyAccumulator:
    """
    Factory function to build the appropriate SaliencyAccumulator subclass based on the configuration's subset selection criteria.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
        input_ids (torch.Tensor): The input token IDs for the batch, used to identify image and generated tokens.
        pixel_values (torch.Tensor | None): The input pixel values for the batch, used for dynamic patch shape inference if needed.
        **kwargs: Additional keyword arguments from the forward pass.
    """

    if isinstance(config.subset_select, (LayerSelect, Sequence)):
        return LayerAccumulator(config, input_ids=input_ids, pixel_values=pixel_values, **kwargs)
    elif isinstance(config.subset_select, (HeadSelect, Mapping)):
        return HeadAccumulator(config, input_ids=input_ids, pixel_values=pixel_values, **kwargs)
    else:
        return SaliencyAccumulator(config, input_ids=input_ids, pixel_values=pixel_values, **kwargs)
