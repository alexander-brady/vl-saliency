from functools import cache

import torch

from vl_saliency.backends.torch.saliency_qk import saliency_qk


@cache
def compiled_saliency_qk():
    return torch.compile(
        saliency_qk,
        mode="max-autotune",
        fullgraph=True,
    )
