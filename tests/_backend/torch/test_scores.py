import pytest
import torch

from vl_saliency._backend.torch.scores import _compute_scores

# -------Reference implementation for testing _compute_scores -----


def manual_scores(q, k, gen_idx, img_idx, scale):
    """
    Slow reference implementation for correctness checking.
    Assumes Hq == Hkv.
    """
    B, H, _, _ = q.shape
    T_gen = gen_idx.shape[1]
    T_img = img_idx.shape[1]

    out = torch.zeros((B, H, T_gen, T_img), dtype=q.dtype)

    for b in range(B):
        for h in range(H):
            for i in range(T_gen):
                qi = q[b, h, gen_idx[b, i]]
                for j in range(T_img):
                    kj = k[b, h, img_idx[b, j]]
                    out[b, h, i, j] = torch.dot(qi, kj)

    return out * scale


# -------Test standard attention (Hq == Hkv) -----


def test_compute_scores_standard_attention_matches_manual():
    torch.manual_seed(0)

    B, H, T, D = 1, 2, 4, 3
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    gen_idx = torch.tensor([[1, 3]])
    img_idx = torch.tensor([[0, 2]])
    scale = 0.5

    scores = _compute_scores(q, k, gen_idx, img_idx, scale)
    expected = manual_scores(q, k, gen_idx, img_idx, scale)

    assert torch.allclose(scores, expected, atol=1e-6)


# -------Behavior tests -----


def test_compute_scores_applies_scale():
    q = torch.ones((1, 1, 2, 2))
    k = torch.ones((1, 1, 2, 2))

    gen_idx = torch.tensor([[0]])
    img_idx = torch.tensor([[1]])

    scores = _compute_scores(q, k, gen_idx, img_idx, scale=2.0)

    # dot([1,1],[1,1]) = 2 → scaled by 2 → 4
    assert scores.item() == 4.0


def test_compute_scores_clamps_negative_indices():
    q = torch.arange(8.0).reshape(1, 1, 4, 2)
    k = torch.arange(8.0).reshape(1, 1, 4, 2)

    # -1 should clamp to 0
    gen_idx = torch.tensor([[-1]])
    img_idx = torch.tensor([[-1]])

    scores = _compute_scores(q, k, gen_idx, img_idx, scale=1.0)

    expected = torch.dot(q[0, 0, 0], k[0, 0, 0])
    assert scores.item() == expected.item()


# -------Test Grouped-query attention (Hq != Hkv) -----


def test_compute_scores_grouped_query_attention():
    torch.manual_seed(0)

    B, Hkv, rep, T, D = 1, 2, 3, 4, 2
    Hq = Hkv * rep

    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)

    gen_idx = torch.tensor([[1, 2]])
    img_idx = torch.tensor([[0, 3]])

    scores = _compute_scores(q, k, gen_idx, img_idx, scale=1.0)

    # Shape check
    assert scores.shape == (B, Hq, 2, 2)

    # Verify correctness head-by-head
    for hq in range(Hq):
        kv_head = hq // rep
        for i in range(2):
            for j in range(2):
                qi = q[0, hq, gen_idx[0, i]]
                kj = k[0, kv_head, img_idx[0, j]]
                expected = torch.dot(qi, kj)
                assert torch.allclose(scores[0, hq, i, j], expected, atol=1e-6)


def test_compute_scores_invalid_grouped_query_raises():
    B, Hq, Hkv, T, D = 1, 3, 2, 4, 2  # 3 not divisible by 2
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)

    gen_idx = torch.tensor([[0]])
    img_idx = torch.tensor([[0]])

    with pytest.raises(AssertionError):
        _compute_scores(q, k, gen_idx, img_idx, scale=1.0)
