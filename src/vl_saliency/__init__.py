"""
Saliency Visualizer for Vision-Language Models using the Attention-Guided CAM (AGCAM) method.

Example usage:
```python
from vl_saliency import SaliencyTrace

model = ...  # Load your Vision Transformer model
processor = ...  # Load your processor for tokenization and image preprocessing

inputs = processor(images=image, text=text, return_tensors="pt")
generated_ids = model(**inputs)

# Initialize SaliencyTrace with the model and processor
trace = SaliencyTrace(model, processor)

# Capture saliency for the generated output
trace.capture(**inputs, generated_ids=generated_ids)

# Compute saliency map for the first token
saliency_map = trace.map(0)
```
"""

from vl_saliency.core.trace import SaliencyTrace

__all__ = ["SaliencyTrace"]
