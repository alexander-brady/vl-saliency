import torch

import vl_saliency.backends.torch.saliency_qk as m


def test_saliency_qk_eager(monkeypatch):
    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8

    # -------mock internals ---
    monkeypatch.setattr(
        m,
        "_compute_scores",
        lambda q, k, gen_idx, img_idx, scale: torch.ones(B, Hq, T_gen, T_img),
    )

    monkeypatch.setattr(
        m,
        "_HEAD_REDUCE",
        {"sum": lambda scores, mask: scores.sum(dim=1)},
    )

    monkeypatch.setattr(
        m,
        "_LAYER_REDUCE",
        {"add": lambda saliency, scores: saliency + scores},
    )

    def dummy_head_op(scores, mask):
        return scores * 2

    def dummy_layer_op(scores, mask):
        return scores * 3

    # -------build function ---
    fn = m.saliency_qk_eager(
        head_reduce="sum",
        layer_reduce="add",
        head_op=dummy_head_op,
        layer_op=dummy_layer_op,
    )

    # -------inputs ---
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    gen_idx = torch.zeros(B, T_gen, dtype=torch.long)
    img_idx = torch.zeros(B, T_img, dtype=torch.long)
    gen_mask = torch.ones(B, T_gen, dtype=torch.bool)
    img_mask = torch.ones(B, T_img, dtype=torch.bool)
    saliency = torch.zeros(B, T_gen, T_img)

    out = fn(
        q,
        k,
        gen_idx,
        gen_mask,
        img_idx,
        img_mask,
        scale=1.0,
        saliency=saliency,
    )

    assert out.shape == (B, T_gen, T_img)
    # ones summed over Hq heads, then multiplied by 2 (head_op) and 3 (layer_op)
    expected = torch.full_like(out, Hq * 2 * 3)
    assert torch.allclose(out, expected)
