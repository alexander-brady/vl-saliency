from __future__ import annotations

from typing import Literal

import torch

from ..core.map import SaliencyMap
from .pipe import Chainable


class SelectLayers(Chainable):
    """Select specific layers from a map.
    Args:
        layers (list[int] | int): List of layer indices to select.
    """

    def __init__(self, layers: list[int] | int):
        self.layers = layers if isinstance(layers, list) else [layers]

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        selected = map.tensor()[self.layers]
        return SaliencyMap(selected)  # shape: [layers, heads, H, W]


class SelectHeads(Chainable):
    """Select specific heads from a map.

    Args:
        heads (list[(int, int)]): List of (layer_index, head_index) tuples to select.
    """

    def __init__(self, heads: list[tuple[int, int]]):
        self.heads = heads

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        layer_idx = [l_idx for l_idx, _ in self.heads]
        head_idx = [h_idx for _, h_idx in self.heads]
        selected = map.tensor()[layer_idx, head_idx]
        selected = selected.unsqueeze(0)  # add layer dim back
        return SaliencyMap(selected)  # shape: [1, num_selected, H, W]


class SelectFirstLayers(Chainable):
    """Select the first N layers from a map.
    Args:
        n (int): Number of layers to select from the start.
    """

    def __init__(self, n: int):
        self.n = n

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        selected = map.tensor()[: self.n]
        return SaliencyMap(selected)  # shape: [layers, heads, H, W]


class SelectLastLayers(Chainable):
    """Select the last N layers from a map.
    Args:
        n (int): Number of layers to select from the end.
    """

    def __init__(self, n: int):
        self.n = n

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        selected = map.tensor()[-self.n :]
        return SaliencyMap(selected)  # shape: [layers, heads, H, W]


class Aggregate(Chainable):
    """Aggregate over layers and heads.

    Args:
        layer_reduce (Literal['mean', 'sum', 'max', 'min', 'prod'], default='mean'): Aggregation method to use.
        head_reduce (Literal['mean', 'sum', 'max', 'min', 'prod'], default='mean'): Aggregation method to use.
    """

    def __init__(
        self,
        layer_reduce: Literal["mean", "sum", "max", "min", "prod"] | None = "mean",
        head_reduce: Literal["mean", "sum", "max", "min", "prod"] | None = "mean",
    ):
        self.layer_reduce = layer_reduce
        self.head_reduce = head_reduce

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        tensor = map.tensor()  # shape: [layers, heads, H, W]

        if self.layer_reduce is not None:
            tensor = self._reduce(tensor, self.layer_reduce, axis=0)

        if self.head_reduce is not None:
            tensor = self._reduce(tensor, self.head_reduce, axis=1)

        # aggregated shape: [1 or layers, 1 or heads, H, W]
        return SaliencyMap(tensor)

    def _reduce(
        self, tensor: torch.Tensor, method: Literal["mean", "sum", "max", "min", "prod"], axis: int
    ) -> torch.Tensor:
        match method:
            case "mean":
                return tensor.mean(dim=axis, keepdim=True)
            case "sum":
                return tensor.sum(dim=axis, keepdim=True)
            case "max":
                return tensor.amax(dim=axis, keepdim=True)
            case "min":
                return tensor.amin(dim=axis, keepdim=True)
            case "prod":
                return tensor.prod(dim=axis, keepdim=True)
