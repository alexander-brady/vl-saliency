from collections.abc import Mapping, Sequence

from jaxtyping import Int
from torch import Tensor

from vl_saliency._core.accum.base import SaliencyAccumulator
from vl_saliency._core.accum.subset import HeadAccumulator, LayerAccumulator
from vl_saliency.config import HeadSelect, LayerSelect, SaliencyConfig


def build_accumulator(
    config: SaliencyConfig,
    input_ids: Int[Tensor, "B S"],
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

    selection = config.selection
    cls: type[SaliencyAccumulator]

    if isinstance(selection, (LayerSelect, Sequence)):
        cls = LayerAccumulator
    elif isinstance(selection, (HeadSelect, Mapping)):
        cls = HeadAccumulator
    else:
        cls = SaliencyAccumulator

    return cls(config, input_ids=input_ids, **kwargs)
