import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image


def overlay(
    saliency_map: torch.tensor,
    image: Image.Image | None = None,
    *,
    ax: plt.Axes | None = None,
    title: str | None = "Saliency Map",
    figsize: tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs,
) -> Figure:
    """
    Visualizes the saliency map on top of the image.

    Args:
        saliency_map (torch.Tensor): The saliency map to visualize. Shape: [1, 1, H, W]
        image (torch.Tensor): The original image. If None, only show the saliency map.
        ax (plt.Axes, optional): Existing axes to draw on. If None, a new Figure is created.
        title (str, optional): Title for the plot. Defaults to "Saliency Map".
        figsize (tuple, optional): Size of the figure. Defaults to (6, 6).
        show_colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
        **plot_kwargs: Additional keyword arguments for the `imshow` function.

    Returns:
        matplotlib.figure.Figure: The figure containing the saliency visualization.
    """
    # Resize and normalize saliency map to [0, 1]
    if image is not None:
        saliency_map = F.interpolate(
            saliency_map, size=image.size[::-1], mode="bilinear", align_corners=False
        )
    saliency_map = saliency_map.detach().squeeze().cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (np.ptp(saliency_map) + 1e-8)

    # Create fig/ax if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot image + saliency
    if image is not None:
        ax.imshow(image)

    defaults = {"cmap": "inferno", "alpha": 0.5, **plot_kwargs}
    im = ax.imshow(saliency_map, **defaults)

    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Attention Weight")

    if title:
        ax.set_title(title)

    ax.axis("off")
    fig.tight_layout()

    return fig
