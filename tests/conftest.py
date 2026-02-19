from typing import cast

import torch
from pytest import fixture

from vl_saliency._types import Backend, Reduction
from vl_saliency.context import SaliencyContext

dummy_backend = cast(Backend, "dummy_backend")


class DummySaliencyContext(SaliencyContext):
    def __init__(
        self,
        attn_implementation: str = "dummy_attn",
        backend: Backend = dummy_backend,
        gen_token_idx: int = 0,
        gen_mask: torch.Tensor | None = None,
        img_token_idx: int = 1,
        img_mask: torch.Tensor | None = None,
        scale: float = 1.0,
        layer_reduce: Reduction = "mean",
        head_reduce: Reduction = "sum",
        saliency: torch.Tensor | None = None,
    ):
        self.attn_implementation = attn_implementation
        self.backend = backend
        self.gen_token_idx = gen_token_idx
        self.gen_mask = gen_mask or torch.tensor([1])
        self.img_token_idx = img_token_idx
        self.img_mask = img_mask or torch.tensor([0])
        self.scale = scale
        self.layer_reduce = layer_reduce
        self.head_reduce = head_reduce
        self.saliency = saliency or torch.tensor([42])
        self.updated_value = None

    def reset(self):
        self.saliency = torch.tensor([42])
        self.updated_value = None

    def update(self, saliency: torch.Tensor):
        self.updated_value = saliency

    def map(self, token: int, batch_idx: int = 0, img_idx: int = 0) -> torch.Tensor:
        return torch.tensor([99.0])


@fixture
def dummy_context():
    return DummySaliencyContext()
