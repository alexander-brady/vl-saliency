from typing import Any, overload

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from PIL.Image import Image

from vl_saliency.ops import Upscale, normalize


@overload
def plot(
    map: Float[torch.Tensor, "H W"],
    image: Image | None = None,
    *,
    ax: None = None,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> Figure: ...


@overload
def plot(
    map: Float[torch.Tensor, "H W"],
    image: Image | None = None,
    *,
    ax: Axes,
    title: str | None = "Saliency Map",
    show_colorbar: bool = True,
    **plot_kwargs,
) -> SubFigure: ...


def plot(
    map: Float[torch.Tensor, "H W"],
    image: Image | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> Figure | SubFigure:
    """Plot saliency map overlaid on an optional image.

    Args:
        saliency_map (torch.Tensor): The saliency map to visualize. Shape: [H, W]
        image (torch.Tensor): The original image. If None, only show the saliency map.
        ax (Axes, optional): Existing axes to draw on. If None, a new Figure is created.
        title (str, optional): Title for the plot. Defaults to "Saliency Map".
        figsize (tuple, optional): Size of the figure. Defaults to (6, 6).
        show_colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
        **plot_kwargs: Additional keyword arguments for the `imshow` function.

    Returns:
        Figure | SubFigure: The figure containing the saliency visualization.
    """

    # Resize and normalize the saliency map to [0, 1]
    if image is not None:
        upscale = Upscale(image.height, image.width, mode="bilinear")
        map = upscale(map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    map = normalize(map)
    map_np = map.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if image is not None:
        ax.imshow(image)

    params: dict[str, Any] = {"cmap": "inferno", "alpha": 0.5, **plot_kwargs}
    im = ax.imshow(map_np, **params)

    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Attention Weight")

    if title:
        ax.set_title(title)

    ax.axis("off")
    return fig
