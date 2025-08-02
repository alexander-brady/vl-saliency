"""
Visualization utilities for saliency maps in vision-language models.

Example usage:
```python
from vl_saliency.utils import visualize_saliency

# Assuming `saliency_map` is a torch.Tensor and `image` is a PIL Image or similar
fig = visualize_saliency(saliency_map, image, title="My Saliency Map", figsize=(8, 8))
fig.show()
```
"""
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np


def visualize_saliency(
    saliency_map,
    image,
    title: Optional[str] = "Saliency Map",
    figsize: Tuple[int, int] = (6, 6),
    show_colorbar: bool = True,
    **plot_kwargs
) -> plt.Figure:
    """
    Visualizes the saliency map on top of the image.
    
    Args:
        saliency_map (torch.Tensor): The saliency map to visualize. Shape: [1, 1, H, W]
        image (torch.Tensor): The original image.
        title (str, optional): Title for the plot. Defaults to "Saliency Map".
        figsize (tuple, optional): Size of the figure. Defaults to (6, 6).
        show_colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
        **plot_kwargs: Additional keyword arguments for the `imshow` function.
        
    Returns:
        matplotlib.figure.Figure: The figure containing the saliency visualization.
    """
    # Resize and normalize saliency map to [0, 1]
    saliency_map = F.interpolate(
        saliency_map, size=image.size[::-1], mode='bilinear', align_corners=False
    )
    saliency_map = saliency_map.detach().squeeze().cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (np.ptp(saliency_map) + 1e-8)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    im = ax.imshow(saliency_map, cmap='coolwarm', alpha=0.5, **plot_kwargs)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='Attention Weight')
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig  