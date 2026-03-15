from jaxtyping import Float, Int
from torch import Tensor


def _compute_scores(
    q: Float[Tensor, "B Hq T D"],
    k: Float[Tensor, "B Hkv T D"],
    gen_idx: Int[Tensor, "B T_gen"],
    img_idx: Int[Tensor, "B T_img"],
    scale: float,
) -> Float[Tensor, "B Hq T_gen T_img"]:
    B, Hq, _, D = q.shape
    Hkv = k.shape[1]

    # Remove -1 indices for gathering
    gen_idx = gen_idx.clamp_min(0)
    img_idx = img_idx.clamp_min(0)

    # Gather the relevant query and key vectors
    qg = q.gather(2, gen_idx[:, None, :, None].expand(B, Hq, -1, D))  # [B, Hq, T_gen, D]
    ki = k.gather(2, img_idx[:, None, :, None].expand(B, Hkv, -1, D))  # [B, Hkv, T_img, D]

    # Compute scaled dot-product scores
    if Hq == Hkv:  # Standard attention
        scores = qg @ ki.transpose(-2, -1)  # [B, Hq, T_gen, T_img]
    else:  # Grouped-query attention
        assert Hq % Hkv == 0
        rep = Hq // Hkv

        # Reshape queries into KV groups
        qg = qg.reshape(B, Hkv, rep, -1, D)  # [B, Hkv, rep, T_gen, D]

        # Compute per-KV-head attention
        scores = qg @ ki.unsqueeze(2).transpose(-2, -1)  # [B, Hkv, rep, T_gen, T_img]

        # Restore head dimension
        scores = scores.reshape(B, Hq, scores.size(-2), scores.size(-1))  # [B, Hq, T_gen, T_img]

    return scores * scale


__all__ = ["_compute_scores"]
