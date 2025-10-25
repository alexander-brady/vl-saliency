from typing import Literal

from ..core.map import SaliencyMap
from .pipe import Chainable


class SelectLayers(Chainable):
    """Select specific layers from a map.
    Args:
        layers (list[int]): List of layer indices to select.
    """

    def __init__(self, layers: list[int]):
        self.layers = layers

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
        return SaliencyMap(selected)  # shape: [layers, heads, H, W]


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
        dim (Literal['layers', 'heads', 'both']): Dimension(s) to aggregate over.
        method (Literal['mean', 'sum', 'max', 'min', 'prod'], default='mean'): Aggregation method to use.
    """

    def __init__(
        self, dim: Literal["layers", "heads", "both"], method: Literal["mean", "sum", "max", "min", "prod"] = "mean"
    ):
        self.dim = dim
        self.method = method

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        tensor = map.tensor()  # shape: [layers, heads, H, W]

        reduced_axis = {"layers": [0], "heads": [1], "both": [0, 1]}[self.dim]

        match self.method:
            case "mean":
                aggregated = tensor.mean(dim=reduced_axis, keepdim=True)
            case "sum":
                aggregated = tensor.sum(dim=reduced_axis, keepdim=True)
            case "max":
                aggregated = tensor.amax(dim=reduced_axis, keepdim=True)
            case "min":
                aggregated = tensor.amin(dim=reduced_axis, keepdim=True)
            case "prod":
                for i in list(sorted(reduced_axis))[::-1]:
                    tensor = tensor.prod(dim=i, keepdim=True)
                aggregated = tensor
            case _:
                raise ValueError(f"Unknown aggregation method: {self.method}")

        # aggregated shape: [1 or layers, 1 or heads, H, W]
        return SaliencyMap(aggregated)
