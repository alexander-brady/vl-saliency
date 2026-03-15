from typing import Literal

import torch

from vl_saliency._backend.triton.kernels import (
    saliency_qk_bwd_dk_kernel,
    saliency_qk_bwd_dq_kernel,
    saliency_qk_fwd_kernel,
)


def _validate_inputs(q, k, gen_idx, gen_mask, img_idx, img_mask):
    assert q.is_cuda and k.is_cuda, "q/k must be CUDA tensors"
    assert q.ndim == 4 and k.ndim == 4, "q/k must be [B, H,T, D]"
    B, Hq, T, D = q.shape
    B2, Hkv, T2, D2 = k.shape
    assert (B2, T2, D2) == (B, T, D)
    assert Hq == Hkv or (Hq % Hkv == 0), "Require Hq==Hkv or Hq%Hkv==0 (GQA)"
    assert gen_idx.shape[0] == B and img_idx.shape[0] == B
    assert gen_mask.shape == gen_idx.shape
    assert img_mask.shape == img_idx.shape
    assert gen_mask.dtype == torch.bool and img_mask.dtype == torch.bool
    assert gen_idx.is_contiguous() and img_idx.is_contiguous()
    assert gen_mask.is_contiguous() and img_mask.is_contiguous()
    assert q.is_contiguous() and k.is_contiguous(), (
        "q/k must be contiguous [B, H, T, D] for best perf"
    )
    return B, Hq, Hkv, T, D, gen_idx.shape[1], img_idx.shape[1]


class SaliencyQKTriton(torch.autograd.Function):
    """Autograd function for saliency QK computation using Triton kernels, handling both forward and backward passes.

    Returns:
        Float[Tensor, "B T_gen T_img"]: Saliency scores for each generated token and image token pair, reduced across heads according to the specified method.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        gen_idx,
        gen_mask,
        img_idx,
        img_mask,
        scale: float,
        head_reduce: Literal["sum", "mean"],
    ):
        B, Hq, Hkv, T, D, Tg, Ti = _validate_inputs(q, k, gen_idx, gen_mask, img_idx, img_mask)

        out = torch.empty((B, Tg, Ti), device=q.device, dtype=torch.float32)

        head_map = {"sum": 0, "mean": 1}
        hr = head_map[head_reduce]

        grid = (B * Tg * Ti,)
        saliency_qk_fwd_kernel[grid](
            q,
            k,
            gen_idx,
            img_idx,
            gen_mask,
            img_mask,
            out,
            B,
            T,
            Tg,
            Ti,
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            scale=torch.tensor(scale, device=q.device, dtype=torch.float32),
            HEAD_REDUCE=hr,
            sqb=q.stride(0),
            sqh=q.stride(1),
            sqt=q.stride(2),
            sqd=q.stride(3),
            skb=k.stride(0),
            skh=k.stride(1),
            skt=k.stride(2),
            skd=k.stride(3),
            sgib=gen_idx.stride(0),
            sgit=gen_idx.stride(1),
            siib=img_idx.stride(0),
            siit=img_idx.stride(1),
            sgmb=gen_mask.stride(0),
            sgmt=gen_mask.stride(1),
            simb=img_mask.stride(0),
            simt=img_mask.stride(1),
            sob=out.stride(0),
            sog=out.stride(1),
            soi=out.stride(2),
        )

        ctx.save_for_backward(q, k, gen_idx, gen_mask, img_idx, img_mask)
        ctx.scale = float(scale)
        ctx.head_reduce = head_reduce
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, gen_idx, gen_mask, img_idx, img_mask = ctx.saved_tensors
        scale = ctx.scale
        hr = ctx.head_reduce

        B, Hq, T, D = q.shape
        Hkv = k.shape[1]
        Tg = gen_idx.shape[1]
        Ti = img_idx.shape[1]

        dq32 = torch.zeros((B, Hq, T, D), device=q.device, dtype=torch.float32)
        dk32 = torch.zeros((B, Hkv, T, D), device=q.device, dtype=torch.float32)

        grid_dq = (B * Hq * Tg,)
        saliency_qk_bwd_dq_kernel[grid_dq](
            k,
            gen_idx,
            img_idx,
            gen_mask,
            img_mask,
            dout,
            dq32,
            B,
            T,
            Tg,
            Ti,
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            scale=torch.tensor(scale, device=q.device, dtype=torch.float32),
            HEAD_REDUCE=hr,
            skb=k.stride(0),
            skh=k.stride(1),
            skt=k.stride(2),
            skd=k.stride(3),
            sgib=gen_idx.stride(0),
            sgit=gen_idx.stride(1),
            siib=img_idx.stride(0),
            siit=img_idx.stride(1),
            sgmb=gen_mask.stride(0),
            sgmt=gen_mask.stride(1),
            simb=img_mask.stride(0),
            simt=img_mask.stride(1),
            sdob=dout.stride(0),
            sdog=dout.stride(1),
            sdoi=dout.stride(2),
            sdqb=dq32.stride(0),
            sdqh=dq32.stride(1),
            sdqt=dq32.stride(2),
            sdqd=dq32.stride(3),
        )

        grid_dk = (B * Hkv * Ti,)
        saliency_qk_bwd_dk_kernel[grid_dk](
            q,
            gen_idx,
            img_idx,
            gen_mask,
            img_mask,
            dout,
            dk32,
            B,
            T,
            Tg,
            Ti,
            Hq=Hq,
            Hkv=Hkv,
            D=D,
            scale=torch.tensor(scale, device=q.device, dtype=torch.float32),
            HEAD_REDUCE=hr,
            sqb=q.stride(0),
            sqh=q.stride(1),
            sqt=q.stride(2),
            sqd=q.stride(3),
            sgib=gen_idx.stride(0),
            sgit=gen_idx.stride(1),
            siib=img_idx.stride(0),
            siit=img_idx.stride(1),
            sgmb=gen_mask.stride(0),
            sgmt=gen_mask.stride(1),
            simb=img_mask.stride(0),
            simt=img_mask.stride(1),
            sdob=dout.stride(0),
            sdog=dout.stride(1),
            sdoi=dout.stride(2),
            sdkb=dk32.stride(0),
            sdkh=dk32.stride(1),
            sdkt=dk32.stride(2),
            sdkd=dk32.stride(3),
        )

        dq = dq32.to(q.dtype)
        dk = dk32.to(k.dtype)

        # Only return gradients for q and k, the rest are None
        return dq, dk, None, None, None, None, None, None
