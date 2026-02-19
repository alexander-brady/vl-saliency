import torch

from vl_saliency._types import Reduction


def saliency_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    gen_idx: torch.Tensor,
    gen_mask: torch.Tensor,
    img_idx: torch.Tensor,
    img_mask: torch.Tensor,
    scale: float,
    layer_reduce: Reduction = "mean",
    head_reduce: Reduction = "mean",
    saliency: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("Triton backend is not yet implemented")
