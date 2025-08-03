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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, HTML
import torch
import torch.nn.functional as F
from transformers import ProcessorMixin


def visualize_saliency(
    saliency_map: torch.tensor,
    image: Image.Image,
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


def render_token_ids(
    token_ids: torch.Tensor,
    processor: ProcessorMixin,
    gen_index: Optional[int] = None,
    skip_tokens: Optional[Union[int, List[int]]] = None
):
    """
    Visualizes the generated text from the model.
    
    Args:
        generated_ids (torch.Tensor): The generated token IDs.
        tokenizer: The tokenizer used to decode the IDs.
        gen_index (Optional[int]): Index from which tokens are considered generated. If None, all tokens are considered prompt. 
        skip_tokens (Optional[Union[int, List[int]]]): Token IDs to skip in the visualization.        
    Returns:
        matplotlib.figure.Figure: The figure containing the generated text visualization.
    """
    token_ids = token_ids.tolist()[0]
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    
    style_block = """
    <style>
    .token-container {
        position: relative;
        display: inline-block;
        padding: 0 2px;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    }

    .token-content {
        display: block;
        border-bottom: 1px solid #999;
        text-align: center;
        transition: border-color 0.2s ease;
    }

    .token-container:hover .token-content {
        border-color: #1e90ff;
    }

    .tooltip {
        visibility: hidden;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 4px;
        padding: 4px 8px;
        position: absolute;
        z-index: 1;
        bottom: 120%;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
        font-size: 14px;
        font-family: sans-serif;
        opacity: 0;
        transition: opacity 0.2s;
    }

    .token-container:hover .tooltip {
        visibility: visible;
        opacity: 1;
    }

    .token-newline {
        display: inline-block;
        opacity: 0.4;
        vertical-align: bottom;
    }

    .token-special {
        display: inline-block;
        opacity: 0.4;
        font-family: monospace;
        margin: 0 2px;
    }

    .token-prefix {
        opacity: 0.4;
    }
    </style>
    """

    html = style_block + "<div>"
    
    skip_tokens = skip_tokens if skip_tokens is not None else []
    if isinstance(skip_tokens, int):
        skip_tokens = [skip_tokens]

    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        if tid in skip_tokens:
            continue
        
        if token in ["\\n", "\n", "Ċ", "▁\n"]:
            html += f'<span class="token-newline">{token}</span><br>'
            continue

        if token.startswith("<") and token.endswith(">"):
            html += f'<span class="token-special">&lt;{token[1:-1]}&gt;</span>'
            continue

        background = "#d5fdd5" if gen_index is not None and i >= gen_index else "#f5f5f5"

        # Show faded leading space token (▁ or Ġ), if present
        if token.startswith(("▁", "Ġ")):
            prefix = token[0]
            body = token[1:]
            display_token = f'<span class="token-prefix">{prefix}</span>{body}'
        else:
            display_token = token

        html += (
            f'<span class="token-container" style="background:{background};">'
            f'<div class="token-content">{display_token}</div>'
            f'<div class="tooltip">Token ID: {tid}</div>'
            f'</span>'
        )

    html += "</div>"
    display(HTML(html))