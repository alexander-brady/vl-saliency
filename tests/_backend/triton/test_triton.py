import pytest
import torch

from vl_saliency._backend.torch import saliency_qk_eager
from vl_saliency._backend.triton import saliency_qk as saliency_qk_triton
from vl_saliency._core.dispatcher import _is_triton_available

from ..utils import dummy_inputs


@pytest.mark.skipif(not _is_triton_available(), reason="Triton not available")
def test_triton_qk():
    # Test that the Triton saliency_qk function can be called and returns a tensor of the expected shape

    B, Hq, Hkv, T, T_gen, T_img, D = 2, 3, 3, 5, 4, 6, 8
    inputs = dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D)

    kwargs = dict(
        head_reduce="sum",
        layer_reduce="sum",
        head_op=None,
        layer_op=None,
    )

    fn_triton = saliency_qk_triton(**kwargs)
    fn_torch = saliency_qk_eager(**kwargs)

    out_triton = fn_triton(**inputs)
    out_torch = fn_torch(**inputs)
    assert out_triton.shape == (B, T_gen, T_img)

    assert torch.allclose(out_triton, out_torch, atol=1e-5)
