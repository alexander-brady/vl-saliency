import pytest
import torch


@pytest.fixture(params=["3d", "4d"])
def sample(request):
    if request.param == "3d":
        x = torch.randn(2, 8, 8)  # (N, H, W)
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        x = torch.randn(2, 3, 8, 8)  # (N, C, H, W)
        mask = torch.ones_like(x, dtype=torch.bool)

    return x, mask
