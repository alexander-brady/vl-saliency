from typing import Any

import torch


def dummy_inputs(B, Hq, Hkv, T, T_gen, T_img, D):
    return dict[str, Any](
        q=torch.randn(B, Hq, T, D),
        k=torch.randn(B, Hkv, T, D),
        gen_idx=torch.zeros(B, T_gen, dtype=torch.long),
        gen_mask=torch.ones(B, T_gen, dtype=torch.bool),
        img_idx=torch.zeros(B, T_img, dtype=torch.long),
        img_mask=torch.ones(B, T_img, dtype=torch.bool),
        scale=1.0,
        saliency=torch.zeros(B, T_gen, T_img),
    )
