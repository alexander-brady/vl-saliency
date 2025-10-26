from typing import overload

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from PIL import Image

from vl_saliency import transforms as T
from vl_saliency.core.map import SaliencyMap


@overload
def overlay(
    saliency_map: SaliencyMap,
    image: Image.Image | None = None,
    *,
    ax: None = None,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> Figure: ...
@overload
def overlay(
    saliency_map: SaliencyMap,
    image: Image.Image | None = None,
    *,
    ax: Axes,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> SubFigure: ...


def overlay(
    saliency_map: SaliencyMap,
    image: Image.Image | None = None,
    *,
    ax: Axes | None = None,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> Figure | SubFigure:
    """
    Visualizes the saliency map on top of the image.

    Args:
        saliency_map (torch.Tensor): The saliency map to visualize. Shape: [1, 1, H, W]
        image (torch.Tensor): The original image. If None, only show the saliency map.
        ax (Axes, optional): Existing axes to draw on. If None, a new Figure is created.
        title (str, optional): Title for the plot. Defaults to "Saliency Map".
        figsize (tuple, optional): Size of the figure. Defaults to (6, 6).
        show_colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
        **plot_kwargs: Additional keyword arguments for the `imshow` function.

    Returns:
        Figure | SubFigure: The figure containing the saliency visualization.
    """
    # Resize and normalize saliency map to [0, 1]
    if image is not None:
        saliency_map >>= T.Upscale(
            x_size=image.width,
            y_size=image.height,
            mode="bilinear",
        )
    saliency_map >>= T.normalize()
    
    # Convert to numpy array for plotting
    tensor = saliency_map.tensor()  # [1, 1, H, W]
    map = tensor.squeeze().cpu().numpy()  # [H, W]

    # Create fig/ax if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot image if provided
    if image is not None:
        ax.imshow(image)

    # Plot saliency map overlay
    params = {"cmap": "inferno", "alpha": 0.5, **plot_kwargs}
    im = ax.imshow(map, **params)

    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Attention Weight")

    if title:
        ax.set_title(title)

    ax.axis("off")
    return fig
