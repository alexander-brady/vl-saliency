
import pytest
import torch

import vl_saliency.backends.torch.saliency_qk as m
from vl_saliency.ops import fusable

from ..utils import dummy_inputs

# ------- Mock Fixtures and Helpers -------


@pytest.fixture(autouse=True)
def mock_saliency_qk_internals(monkeypatch):
    # Mock the internal functions to isolate testing of saliency_qk_eager and saliency_qk_compiled
    monkeypatch.setattr(m, "_compute_scores", lambda *args, **kwargs: torch.ones(2, 3, 4, 6))
    monkeypatch.setattr(m, "_HEAD_REDUCE", {"sum": lambda scores, mask: scores.sum(dim=1)})
    monkeypatch.setattr(m, "_LAYER_REDUCE", {"add": lambda saliency, scores: saliency + scores})


@pytest.fixture
def mock_compile(monkeypatch):
    called = {}

    def mock_compile(fn, *args, **kwargs):
        called.update(kwargs)
        return fn  # Return the original function without compilation

    monkeypatch.setattr(torch, "compile", mock_compile)
    return called


def dummy_head_op(scores, mask):  # Not marked fuseable
    return scores * 2


def dummy_layer_op(scores, mask):  # Not marked fuseable
    return scores * 3


# ------- Eager Tests -------


def test_saliency_qk_eager(monkeypatch):
    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8

    # -------build function ---
    fn = m.saliency_qk_eager(
        head_reduce="sum",
        layer_reduce="add",
        head_op=dummy_head_op,
        layer_op=dummy_layer_op,
    )

    inputs = dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D)
    out = fn(**inputs)

    assert out.shape == (B, T_gen, T_img)
    # ones summed over Hq heads, then multiplied by 2 (head_op) and 3 (layer_op)
    expected = torch.full_like(out, Hq * 2 * 3)
    assert torch.allclose(out, expected)


# ------- Compiled Tests -------


@pytest.mark.parametrize("fuse_head_op", [True, False])
@pytest.mark.parametrize("fuse_layer_op", [True, False])
def test_saliency_qk_compiled_full_graph(mock_compile, fuse_head_op, fuse_layer_op):

    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8
    inputs = dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D)

    head_op = dummy_head_op if fuse_head_op else fusable(dummy_head_op)
    layer_op = dummy_layer_op if fuse_layer_op else fusable(dummy_layer_op)

    fn = m.saliency_qk_compiled(
        head_reduce="sum",
        layer_reduce="add",
        head_op=head_op,
        layer_op=layer_op,
    )

    eager_fn = m.saliency_qk_eager(
        head_reduce="sum",
        layer_reduce="add",
        head_op=head_op,
        layer_op=layer_op,
    )

    assert mock_compile.get("fullgraph") == (not fuse_head_op and not fuse_layer_op)
    assert torch.allclose(fn(**inputs), eager_fn(**inputs))


def test_saliency_qk_no_ops(mock_compile):
    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8
    inputs = dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D)

    fn = m.saliency_qk_eager(
        head_reduce="sum",
        layer_reduce="add",
        head_op=None,
        layer_op=None,
    )

    out = fn(**inputs)
    assert out.shape == (B, T_gen, T_img)
    expected = torch.full_like(out, Hq)  # ones summed over Hq heads
    assert torch.allclose(out, expected)

    compiled_fn = m.saliency_qk_compiled(
        head_reduce="sum",
        layer_reduce="add",
        head_op=None,
        layer_op=None,
    )

    assert torch.allclose(compiled_fn(**inputs), out)
    assert mock_compile.get("fullgraph")  # Should be fully fused when no ops are present


# ------- Fusion Tests -------


def compile_works():
    return hasattr(torch, "compile") and hasattr(torch, "cuda") and torch.cuda.is_available()


@pytest.mark.skipif(
    not compile_works(),
    reason="torch.compile not supported in this environment",
)
def test_saliency_qk_compiled_fusion():
    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8
    inputs = dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D)

    # Use fusable ops to allow full fusion
    head_op = fusable(dummy_head_op)
    layer_op = fusable(dummy_layer_op)

    fn = m.saliency_qk_compiled(
        head_reduce="sum",
        layer_reduce="add",
        head_op=head_op,
        layer_op=layer_op,
    )

    eager_fn = m.saliency_qk_eager(
        head_reduce="sum",
        layer_reduce="add",
        head_op=head_op,
        layer_op=layer_op,
    )

    assert torch.allclose(fn(**inputs), eager_fn(**inputs))
