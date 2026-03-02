from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

from jaxtyping import Float, Int
from torch import Tensor
from transformers.utils.generic import ModelOutput

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.api.out import SaliencyOutput
from vl_saliency.core.accum import build_accumulator


def build_saliency_forward(
    config: SaliencyConfig,
    attn_implementation: str,
    forward: Callable[..., ModelOutput],
) -> Callable[..., SaliencyOutput]:
    """Builds a custom forward method that creates a SaliencyTrace and passes it through the model's forward pass to compute saliency maps."""

    @overload
    def _saliency_forward(
        model_self,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        *,
        return_dict: Literal[True] | None = None,
        **kwargs,
    ) -> SaliencyOutput: ...

    @overload
    def _saliency_forward(
        model_self,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        *,
        return_dict: Literal[False],
        **kwargs,
    ) -> tuple[Any, ...]: ...

    def _saliency_forward(
        model_self,
        input_ids: Int[Tensor, "B S"],
        pixel_values: Float[Tensor, "B C H W"] | None = None,
        *,
        return_dict: bool | None = None,
        **kwargs,
    ) -> SaliencyOutput | tuple[Any, ...]:

        trace = kwargs.get("saliency") or build_accumulator(
            config=config,
            input_ids=input_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        kwargs["saliency"] = trace
        kwargs.setdefault("attn_implementation", attn_implementation)

        out = forward(model_self, input_ids=input_ids, pixel_values=pixel_values, **kwargs)

        default_return_dict = (
            model_self.config.return_dict if hasattr(model_self, "config") else True
        )
        return_dict = return_dict if return_dict is not None else default_return_dict
        if not return_dict and not isinstance(out, tuple):
            out = out.to_tuple()

        if isinstance(out, tuple):  # return_dict=False
            out = tuple(list(out) + [trace.saliency])
        else:
            out = SaliencyOutput(base_output=out, saliency=trace.saliency)
        return out

    return _saliency_forward
