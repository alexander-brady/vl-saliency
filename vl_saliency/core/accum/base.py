from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor

from vl_saliency.api.config import SaliencyConfig
from vl_saliency.backends.dispatcher import assign_auto, get_qk_accumulator
from vl_saliency.core.grid import SaliencyGrid
from vl_saliency.core.layout import SequenceLayout
from vl_saliency.types import Reduction
from vl_saliency.utils.logging import get_logger

logger = get_logger(__name__)


class SaliencyAccumulator:
    """
    Holds necessary information for saliency extraction during the forward pass, including token layout and accumulated saliency map.

    Args:
        config (SaliencyConfig): The configuration object containing parameters for saliency extraction.
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
        self.layout = SequenceLayout(
            config, input_ids=input_ids, pixel_values=pixel_values, **kwargs
        )

        self.attn_scale = config.attn_scale
        self.layer_reduce: Reduction = config.layer_reduce
        self.layers_accumulated = 0

        self._resolve_qk_fn(config)

        self._saliency: Float[Tensor, "B T_gen T_img"]
        self._init_saliency(
            shape=(self.layout.B, self.layout.T_gen, self.layout.T_img),
            device=input_ids.device,
            dtype=torch.float32,
        )

    @property
    def saliency(self) -> SaliencyGrid:
        """Return the stored saliency map tensor."""
        tensor = self._saliency
        if self.layer_reduce == "mean" and self.layers_accumulated > 0:
            tensor = tensor / self.layers_accumulated
        return SaliencyGrid(tensor, self.layout)

    def accumulate_qk(self, q: Float[Tensor, "B Hq T D"], k: Float[Tensor, "B Hkv T D"]):
        """Accumulates saliency contributions from the given query and key tensors for the current layer, updating the map."""
        self.layers_accumulated += 1
        self._saliency = self._qk_accumulator(
            q,
            k,
            gen_idx=self.layout.gen_token_idx,
            gen_mask=self.layout.gen_mask,
            img_idx=self.layout.img_token_idx,
            img_mask=self.layout.img_mask,
            scale=self.attn_scale,
            saliency=self._saliency,
        )

    def _init_saliency(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype):
        """Initializes the saliency map tensor based on the specified reduction method."""
        match self.layer_reduce:
            case "mean" | "sum":
                self._saliency = torch.zeros(shape, device=device, dtype=dtype)
            case "max":
                self._saliency = torch.full(shape, float("-inf"), device=device, dtype=dtype)
            case "min":
                self._saliency = torch.full(shape, float("inf"), device=device, dtype=dtype)
            case "prod":
                self._saliency = torch.ones(shape, device=device, dtype=dtype)

    def _resolve_qk_fn(self, config: SaliencyConfig):
        """Initializes the backend function for saliency accumulation."""
        if config.backend == "auto":
            backend = assign_auto(self.layout.device, head_reduce=config.head_reduce)
            logger.info_once(f"Auto-assigned backend '{backend}'.")
        else:
            backend = config.backend

        self._qk_accumulator = get_qk_accumulator(
            backend=backend,
            head_reduce=config.head_reduce,
            layer_reduce=config.layer_reduce,
            head_op=config.head_op,
            layer_op=config.layer_op,
        )
