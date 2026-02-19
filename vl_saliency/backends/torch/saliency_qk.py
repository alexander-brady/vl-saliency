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
    B, Hq, T, D = q.shape
    Hkv = k.shape[1]

    if Hq != Hkv:
        assert Hq % Hkv == 0
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)  # [B, Hq, T, D]

    # Remove -1 indices for gathering
    gen_idx = gen_idx.clamp(min=0)
    img_idx = img_idx.clamp(min=0)

    # Gather the relevant query and key vectors
    qg = q.gather(dim=2, index=gen_idx[:, None, :, None].expand(B, Hq, -1, D))  # [B, Hq, T_gen, D]
    ki = k.gather(dim=2, index=img_idx[:, None, :, None].expand(B, Hq, -1, D))  # [B, Hq, T_img, D]

    # Compute scaled dot product attention scores: [B, Hq, T_gen, T_img]
    scores = torch.einsum("b h t d, b h s d -> b h t s", qg, ki) * scale

    # Mask out padding tokens
    scores = scores * gen_mask[:, None, :, None] * img_mask[:, None, None, :]

    # Reduce over heads and layers
    match head_reduce:
        case "mean":
            scores = scores.mean(dim=1)
        case "sum":
            scores = scores.sum(dim=1)
        case "max":
            scores = scores.max(dim=1).values
        case "min":
            scores = scores.min(dim=1).values
        case "prod":
            scores = scores.prod(dim=1)

    # Aggregate with existing saliency
    match layer_reduce:
        case "mean" | "sum":
            saliency = saliency + scores
        case "max":
            saliency = torch.max(saliency, scores)
        case "min":
            saliency = torch.min(saliency, scores)
        case "prod":
            saliency = saliency * scores

    return saliency
