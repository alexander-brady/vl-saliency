import triton
import triton.language as tl

# -------- Forward: out[B, Tg, Ti] -------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_H": 4}, num_warps=4),
        triton.Config({"BLOCK_D": 64, "BLOCK_H": 8}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_H": 4}, num_warps=8),
        triton.Config({"BLOCK_D": 128, "BLOCK_H": 8}, num_warps=8),
    ],
    key=["D", "Hq"],
)
@triton.jit
def saliency_qk_fwd_kernel(
    Q_ptr,
    K_ptr,
    GEN_IDX_ptr,
    IMG_IDX_ptr,
    GEN_MASK_ptr,
    IMG_MASK_ptr,
    OUT_ptr,
    # runtime sizes
    B: tl.int32,
    T: tl.int32,
    Tg: tl.int32,
    Ti: tl.int32,
    # compile-time (model-fixed) sizes
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    scale,  # float32 runtime scalar
    HEAD_REDUCE: tl.constexpr,  # 0=sum, 1=mean
    # strides (elements)
    sqb: tl.constexpr,
    sqh: tl.constexpr,
    sqt: tl.constexpr,
    sqd: tl.constexpr,
    skb: tl.constexpr,
    skh: tl.constexpr,
    skt: tl.constexpr,
    skd: tl.constexpr,
    sgib: tl.constexpr,
    sgit: tl.constexpr,
    siib: tl.constexpr,
    siit: tl.constexpr,
    sgmb: tl.constexpr,
    sgmt: tl.constexpr,
    simb: tl.constexpr,
    simt: tl.constexpr,
    sob: tl.constexpr,
    sog: tl.constexpr,
    soi: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)

    tgti = Tg * Ti
    b = pid // tgti
    rem = pid - b * tgti
    tg = rem // Ti
    ti = rem - tg * Ti

    inb = (b < B) & (tg < Tg) & (ti < Ti)

    gen_idx = tl.load(GEN_IDX_ptr + b * sgib + tg * sgit, mask=inb, other=0.0).to(tl.int32)
    img_idx = tl.load(IMG_IDX_ptr + b * siib + ti * siit, mask=inb, other=0.0).to(tl.int32)
    gen_idx = tl.maximum(gen_idx)
    img_idx = tl.maximum(img_idx)

    gen_ok = tl.load(GEN_MASK_ptr + b * sgmb + tg * sgmt, mask=inb, other=0).to(tl.int1)
    img_ok = tl.load(IMG_MASK_ptr + b * simb + ti * simt, mask=inb, other=0).to(tl.int1)
    valid = gen_ok & img_ok & inb

    use_gqa = Hq != Hkv
    rep = Hq // Hkv  # 1 if no GQA

    offs_d = tl.arange(0, BLOCK_D)
    head_sum = tl.zeros((), dtype=tl.float32)

    # Sum over heads in blocks
    for h0 in tl.static_range(0, Hq, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        hmask = offs_h < Hq

        offs_hkv = tl.where(use_gqa, offs_h // rep, offs_h).to(tl.int32)

        acc_h = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for d0 in tl.static_range(0, D, BLOCK_D):
            d_mask = (d0 + offs_d) < D
            q_ptrs = Q_ptr + b * sqb + offs_h[:, None] * sqh + gen_idx * sqt + (d0 + offs_d) * sqd
            k_ptrs = K_ptr + b * skb + offs_hkv[:, None] * skh + img_idx * skt + (d0 + offs_d) * skd

            q = tl.load(q_ptrs, mask=hmask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            k = tl.load(k_ptrs, mask=hmask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            acc_h += tl.sum(q * k, axis=1)

        acc_h *= scale
        acc_h = tl.where(valid & hmask, acc_h, 0.0)
        head_sum += tl.sum(acc_h, axis=0)

    if HEAD_REDUCE == 1:  # mean
        head_sum = head_sum * (1.0 / Hq)

    out_ptr = OUT_ptr + b * sob + tg * sog + ti * soi
    tl.store(out_ptr, head_sum)


# ------- Backward dQ: dq[B, Hq, T, D] fp32, grid: (B, Hq, Tg)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_I": 32}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_I": 32}, num_warps=8),
        triton.Config({"BLOCK_D": 64, "BLOCK_I": 64}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def saliency_qk_bwd_dq_kernel(
    K_ptr,
    GEN_IDX_ptr,
    IMG_IDX_ptr,
    GEN_MASK_ptr,
    IMG_MASK_ptr,
    dOUT_ptr,
    dQ_ptr,
    # runtime sizes
    B: tl.int32,
    T: tl.int32,
    Tg: tl.int32,
    Ti: tl.int32,
    HEAD_REDUCE: tl.constexpr,  # 0=sum, 1=mean
    # compile-time sizes
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    scale,  # float32 runtime scalar
    # strides
    skb: tl.constexpr,
    skh: tl.constexpr,
    skt: tl.constexpr,
    skd: tl.constexpr,
    sgib: tl.constexpr,
    sgit: tl.constexpr,
    siib: tl.constexpr,
    siit: tl.constexpr,
    sgmb: tl.constexpr,
    sgmt: tl.constexpr,
    simb: tl.constexpr,
    simt: tl.constexpr,
    sdob: tl.constexpr,
    sdog: tl.constexpr,
    sdoi: tl.constexpr,
    sdqb: tl.constexpr,
    sdqh: tl.constexpr,
    sdqt: tl.constexpr,
    sdqd: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid = tl.program_id(0)

    per_b = Hq * Tg
    b = pid // per_b
    rem = pid - b * per_b
    hq = rem // Tg
    tg = rem - hq * Tg

    inb = (b < B) & (hq < Hq) & (tg < Tg)

    gen_idx = tl.load(GEN_IDX_ptr + b * sgib + tg * sgit, mask=inb, other=0).to(tl.int32)
    gen_idx = tl.maximum(gen_idx)
    gen_ok = tl.load(GEN_MASK_ptr + b * sgmb + tg * sgmt, mask=inb, other=0).to(tl.int1)

    use_gqa = Hq != Hkv
    rep = Hq // Hkv
    hkv = tl.where(use_gqa, hq // rep, hq).to(tl.int32)

    offs_d = tl.arange(0, BLOCK_D)
    dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over image tokens
    for i0 in tl.static_range(
        0, 65536, step=BLOCK_I
    ):  # static_range needs a constant upper; we guard with mask
        offs_i = i0 + tl.arange(0, BLOCK_I)
        imask = offs_i < Ti
        if not any(imask):
            break

        img_idx = tl.load(IMG_IDX_ptr + b * siib + offs_i * siit, mask=inb & imask, other=0).to(
            tl.int32
        )
        img_idx = tl.maximum(img_idx)
        img_ok = tl.load(IMG_MASK_ptr + b * simb + offs_i * simt, mask=inb & imask, other=0).to(
            tl.int1
        )

        valid = inb & gen_ok & img_ok & imask

        grad = tl.load(dOUT_ptr + b * sdob + tg * sdog + offs_i * sdoi, mask=valid, other=0.0).to(
            tl.float32
        )
        grad = grad * scale
        if HEAD_REDUCE == 1:  # mean
            grad = grad * (1.0 / Hq)

        k_ptrs = K_ptr + b * skb + hkv * skh + img_idx[:, None] * skt + offs_d[None, :] * skd
        k = tl.load(k_ptrs, mask=valid[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        dq_acc += tl.sum(k * grad[:, None], axis=0)

    dq_ptr = dQ_ptr + b * sdqb + hq * sdqh + offs_d * sdqd
    out = tl.where(inb & gen_ok, dq_acc, 0.0)

    tl.store(dq_ptr, out, mask=inb & (offs_d < D))


# ------- Backward dK: dk[B, Hkv, T, D] fp32, grid: (B, Hkv, Ti) -------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64, "BLOCK_G": 16}, num_warps=4),
        triton.Config({"BLOCK_D": 128, "BLOCK_G": 16}, num_warps=8),
        triton.Config({"BLOCK_D": 64, "BLOCK_G": 32}, num_warps=8),
    ],
    key=["D"],
)
@triton.jit
def saliency_qk_bwd_dk_kernel(
    Q_ptr,
    GEN_IDX_ptr,
    IMG_IDX_ptr,
    GEN_MASK_ptr,
    IMG_MASK_ptr,
    dOUT_ptr,
    dK_ptr,
    # runtime sizes
    B: tl.int32,
    T: tl.int32,
    Tg: tl.int32,
    Ti: tl.int32,
    # compile-time sizes
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,
    scale,  # float32 runtime scalar
    HEAD_REDUCE: tl.constexpr,  # 0=sum, 1=mean
    # strides
    sqb: tl.constexpr,
    sqh: tl.constexpr,
    sqt: tl.constexpr,
    sqd: tl.constexpr,
    sgib: tl.constexpr,
    sgit: tl.constexpr,
    siib: tl.constexpr,
    siit: tl.constexpr,
    sgmb: tl.constexpr,
    sgmt: tl.constexpr,
    simb: tl.constexpr,
    simt: tl.constexpr,
    sdob: tl.constexpr,
    sdog: tl.constexpr,
    sdoi: tl.constexpr,
    sdkb: tl.constexpr,
    sdkh: tl.constexpr,
    sdkt: tl.constexpr,
    sdkd: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0)
    per_b = Hkv * Ti
    b = pid // per_b
    rem = pid - b * per_b
    hkv = rem // Ti
    ti = rem - hkv * Ti

    inb = (b < B) & (hkv < Hkv) & (ti < Ti)

    img_idx = tl.load(IMG_IDX_ptr + b * siib + ti * siit, mask=inb, other=0).to(tl.int32)
    img_idx = tl.maximum(img_idx)
    img_ok = tl.load(IMG_MASK_ptr + b * simb + ti * simt, mask=inb, other=0).to(tl.int1)

    rep = Hq // Hkv
    hq0 = hkv * rep

    offs_d = tl.arange(0, BLOCK_D)
    dk_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for g0 in tl.static_range(0, 65536, BLOCK_G):
        offs_g = g0 + tl.arange(0, BLOCK_G)
        gmask = offs_g < Tg
        if not any(gmask):
            break

        gen_idx = tl.load(GEN_IDX_ptr + b * sgib + offs_g * sgit, mask=inb & gmask, other=0).to(
            tl.int32
        )
        gen_idx = tl.maximum(gen_idx)
        gen_ok = tl.load(GEN_MASK_ptr + b * sgmb + offs_g * sgmt, mask=inb & gmask, other=0).to(
            tl.int1
        )

        valid = inb & img_ok & gen_ok & gmask

        grad = tl.load(dOUT_ptr + b * sdob + offs_g * sdog + ti * sdoi, mask=valid, other=0.0).to(
            tl.float32
        )
        grad = grad * scale
        if HEAD_REDUCE == 1:  # mean
            grad = grad * (1.0 / Hq)

        # Sum across the query heads mapped to this kv head
        for r in tl.static_range(rep):
            hq = hq0 + r
            q_ptrs = Q_ptr + b * sqb + hq * sqh + gen_idx[:, None] * sqt + offs_d[None, :] * sqd
            q = tl.load(q_ptrs, mask=valid[:, None] & (offs_d[None, :] < D), other=0.0).to(
                tl.float32
            )
            dk_acc += tl.sum(q * grad[:, None], axis=0)

        dk_ptr = dK_ptr + b * sdkb + hkv * sdkh + img_idx * sdkt + offs_d * sdkd
        out = tl.where(inb & img_ok, dk_acc, 0.0)
        tl.store(dk_ptr, out, mask=inb & (offs_d < D))
