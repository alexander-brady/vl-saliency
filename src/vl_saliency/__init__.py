"""
Saliency Visualizer for Vision-Language Models using the Attention-Guided CAM (AGCAM) method.

Example usage:
```python
from vl_saliency import AGCAM

model = ...  # Load your Vision Transformer model
processor = ...  # Load your processor for tokenization and image preprocessing
inputs = processor(images=image, text=text, return_tensors="pt")

# Initialize AGCAM with the model and processor
agcam = AGCAM(model, pad_token_id=processor.tokenizer.pad_token_id)
generated_ids, saliency = agcam.generate(**inputs)
```
"""
from vl_saliency.AGCAM import AGCAM