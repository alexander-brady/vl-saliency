"""
Saliency Visualizer for Vision-Language Models using the Attention-Guided CAM (AGCAM) method.

Example usage:
```python
from vl_saliency import SaliencyExtractor

model = ...  # Load your Vision Transformer model
processor = ...  # Load your processor for tokenization and image preprocessing
inputs = processor(images=image, text=text, return_tensors="pt")

# Initialize SaliencyExtractor with the model and processor
extractor = SaliencyExtractor(model, processor)

# Generate model output and saliency map for the first token
generated_ids = extractor.generate(**inputs)
saliency_map = extractor.compute_saliency(0)
```
"""
from vl_saliency.extractor import SaliencyExtractor