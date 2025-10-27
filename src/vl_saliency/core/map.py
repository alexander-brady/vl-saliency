from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL.Image import Image

if TYPE_CHECKING:
    from ..transforms.layers import reduction
    from ..transforms.pipe import Transform

_ALLOWED_TORCH_FUNCTIONS = frozenset(
    {
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        torch.equal,
        torch.ne,
        torch.clamp,
        torch.where,
        torch.minimum,
        torch.maximum,
        torch.pow,
        torch.mean,
        torch.sum,
        torch.abs,
        torch.sqrt,
        torch.exp,
        torch.log,
        torch.sigmoid,
        torch.relu,
    }
)


class SaliencyMap:
    """
    A saliency map object representing the importance of image regions.

    Arguments:
        map (torch.Tensor): A 4D tensor of shape [layers, heads, H, W] representing the saliency map.

    Methods:
        apply(transform) -> SaliencyMap: Apply a Transform or Pipeline to the saliency map (can also use >>)
        tensor() -> torch.Tensor: Return a clone of the underlying tensor.
        numpy() -> np.ndarray: Return a NumPy array representation of the underlying tensor.
        save(path, **kwargs): Save the saliency map tensor to a file using torch.save
        plot(image=None, show=True, **kwargs) -> Figure: Plot the saliency map overlaid on an optional image.
        Arithmetic operations: +, -, *, / with other SaliencyMap or scalars.
        Torch functions: Supports a limited set of torch functions via __torch_function__.
    """

    def __init__(self, map: torch.Tensor):
        if not map.dim() == 4:  # [layers, heads, H, W]
            raise ValueError("SaliencyMap must have 4 dimensions [layers, heads, H, W].")
        self.map = map.detach().cpu().clone()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, SaliencyMap):
            return False
        return torch.equal(self.map, value.map)

    @staticmethod
    def _unwrap(x):
        return x.map if isinstance(x, SaliencyMap) else x

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func not in _ALLOWED_TORCH_FUNCTIONS:
            raise NotImplementedError("Unsupported torch function for SaliencyMap.")

        kwargs = kwargs or {}

        # immutability: no in-place or out=
        name = getattr(func, "__name__", "")
        if name.endswith("_") or "out" in kwargs:
            raise TypeError("SaliencyMap is immutable; in-place/out= ops are not allowed.")

        uargs = tuple(cls._unwrap(a) for a in args)
        ukw = {k: cls._unwrap(v) for k, v in kwargs.items()}
        res = func(*uargs, **ukw)

        # wrap only if it’s a 4D tensor again
        return SaliencyMap(res) if isinstance(res, torch.Tensor) and res.dim() == 4 else res

    def __add__(self, other) -> SaliencyMap:
        return SaliencyMap(torch.add(self._unwrap(self), self._unwrap(other)))

    def __sub__(self, other) -> SaliencyMap:
        return SaliencyMap(torch.sub(self._unwrap(self), self._unwrap(other)))

    def __mul__(self, other) -> SaliencyMap:
        return SaliencyMap(torch.mul(self._unwrap(self), self._unwrap(other)))

    def __truediv__(self, other) -> SaliencyMap:
        return SaliencyMap(torch.div(self._unwrap(self), self._unwrap(other)))

    def __rshift__(self, other: Transform) -> SaliencyMap:
        return self.apply(other)

    def apply(self, transform: Transform) -> SaliencyMap:
        """Apply a Transform, returning a new SaliencyMap."""
        map = SaliencyMap(self.map)  # Create a new SaliencyMap with cloned tensor
        out = transform(map)
        return out

    def tensor(self) -> torch.Tensor:
        """Return a clone of the underlying tensor."""
        return self.map.clone()

    def numpy(self) -> np.ndarray:
        """Return a NumPy array representation of the underlying tensor."""
        return self.tensor().numpy()

    def save(self, path: str, **kwargs):
        """Save the saliency map tensor to a file using torch.save."""
        torch.save(self.map, path, **kwargs)

    def agg(
        self,
        layer_reduce: reduction | None = "mean",
        head_reduce: reduction | None = "mean",
    ) -> SaliencyMap:
        """Aggregate the saliency map along specified dimensions.

        Args:
            layer_reduce (Literal['mean', 'sum', 'max', 'min', 'prod'] | None, default='mean'):
                Aggregation method for layers. If None, no aggregation is performed on layers.
            head_reduce (Literal['mean', 'sum', 'max', 'min', 'prod'] | None, default='mean'):
                Aggregation method for heads. If None, no aggregation is performed on heads.

        Returns:
            SaliencyMap: The aggregated saliency map.
        """
        from ..transforms import Aggregate

        return self >> Aggregate(layer_reduce=layer_reduce, head_reduce=head_reduce)

    def plot(
        self,
        image: Image | None = None,
        show: bool = True,
        *,
        title: str | None = "Saliency Map",
        figsize: tuple[int, int] = (6, 6),
        colorbar: bool = True,
        **plot_kwargs,
    ) -> Figure:
        """
        Plot the saliency map overlaid on an optional image.

        Args:
            image (PIL.Image.Image | None, default=None): An optional image to overlay the saliency map on.
            show (bool, default=True): Whether to display the plot immediately.
            title (str | None, default="Saliency Map"): The title of the plot.
            figsize (tuple[int, int], default=(6, 6)): The size of the figure.
            colorbar (bool, default=True): Whether to show a colorbar alongside the plot.
            **plot_kwargs: Additional keyword arguments passed to the overlay function.

        Returns:
            matplotlib.figure.Figure: The resulting figure object.
        """

        # Ensure shape is [1, 1, H, W]
        layers, heads, _, _ = self.map.shape
        if layers > 1 or heads > 1:
            raise ValueError(
                "Plotting is only supported for saliency maps with a single layer and head. "
                f"Current shape: [{layers}, {heads}, H, W]. Please `agg()` first."
            )

        # Prepare overlay arguments
        plot_kwargs.pop("ax", None)
        plot_kwargs["show_colorbar"] = colorbar
        plot_kwargs["title"] = title
        plot_kwargs["figsize"] = figsize

        from ..viz.overlay import overlay

        fig = overlay(self, image=image, **plot_kwargs)

        if show:
            fig.show()
        return fig
