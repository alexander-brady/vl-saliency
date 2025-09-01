"""
Visualization utilities for saliency maps in vision-language models.

Example usage:
```python
from vl_saliency.viz import overlay

# Assuming `saliency_map` is a torch.Tensor and `image` is a PIL Image or similar
fig = overlay(saliency_map, image, title="My Saliency Map", figsize=(8, 8))
fig.show()
```
"""

from .overlay import overlay

__all__ = ["overlay"]
