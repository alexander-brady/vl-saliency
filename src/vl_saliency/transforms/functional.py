import numpy as np
import torch

from ..core.map import SaliencyMap
from . import chainable


@chainable
def relu(map: SaliencyMap) -> SaliencyMap:
    """Apply ReLU activation to a SaliencyMap."""
    return SaliencyMap(torch.relu(map.tensor()))


@chainable
def abs(map: SaliencyMap) -> SaliencyMap:
    """Apply absolute values to SaliencyMap."""
    return SaliencyMap(torch.abs(map.tensor()))


@chainable
def sigmoid(map: SaliencyMap) -> SaliencyMap:
    """Apply the sigmoid transformation to a SaliencyMap."""
    return SaliencyMap(torch.sigmoid(map.tensor()))


@chainable
def normalize(map: SaliencyMap) -> SaliencyMap:
    """Normalize a saliency map to the range [0, 1]."""
    tensor = map.tensor()
    map_np = tensor.detach().cpu().numpy()
    normalized = (map_np - map_np.min()) / (np.ptp(map_np) + 1e-8)
    tensor = torch.from_numpy(normalized).to(device=tensor.device, dtype=tensor.dtype)
    return SaliencyMap(tensor)
