# Vision-Language Saliency Extraction

This repository provides tools for extracting saliency maps from vision-language models, extending classic methods originally developed for Vision Transformers (ViTs) to modern vision-language architectures.


**Project Structure**

- `src/vl_saliency/`: Contains the main code for saliency extraction.
- `notebooks/`: Example notebooks demonstrating how to use the saliency extractor with different vision-language models.
- `scripts/`: Shell scripts for starting Jupyter servers on the [ETH Zurich Euler cluster](https://scicomp.ethz.ch/wiki/Euler).

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the required packages:
   ```bash
    pip install -e .[dev] # Install the package in editable mode
    ```
    
## Usage

> See the [example notebook](notebooks/quickstart.ipynb) for a complete example of how to use the saliency extractor with a Gemma3 vision-language model.

This library is compatible with any Huggingface Image-Text-to-Text model.

Quickstart code snippet:

```python
from vl_saliency import SaliencyTrace
from vl_saliency.viz import overlay

# Initialize the model and input prompt
model = AutoModel.from_pretrained("model_name")  # Replace with your model name
processor = AutoProcessor.from_pretrained("model_name")  # Replace with your processor name

image = PIL.Image.open("path_to_image.jpg")  # Load your image
inputs = processor(text="Your prompt", images=image, return_tensors="pt")

# Initialize the saliency extractor
trace = SaliencyTrace(model, processor)

# Generate response 
with torch.inference_mode():
    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=200) 
    
# Compute attention and gradients
trace.capture(**inputs, generated_ids=generated_ids, visualize_tokens=True) 

# Compute the saliency map using the AGCAM (default) algorithm
saliency_map = trace.compute_saliency(token=200)  # Change token_index as needed

# Visualize the saliency map
fig = overlay(saliency_map, image, title="Saliency Map")
fig.show()  # or save the plot using fig.savefig("saliency_map.png")
```

## Saliency Computations

> See the [variations notebook](notebooks/variations.ipynb) for a complete example of different saliency computation techniques available.

The module is easily customizable, allowing for different saliency computation methodologies. Default parameters can be set in the `SaliencyTrace` constructor, and can be overridden on a per-call basis. The following parameters can be adjusted:

- `layers`: Specifies the attention layers to include in the saliency computation (e.g., `[-1]` for the last layer only). Default: `ALL_LAYERS`
- `layer_reduce`: Specifies the reduction method for layers (e.g., `mean`, `max`). Default: `sum`
- `head_reduce`: Specifies the reduction method for heads (e.g., `mean`, `max`). Default: `sum`
- `method`: Specifies the saliency computation method (e.g., `AGCAM`, `GRADCAM`). Default: `AGCAM`. See below.

**Method**

This can be a string or a callable. If callable, it must accept the attention and gradient tensors and return a mask tensor of the same shape.

Additional kwargs can be passed to the saliency computation methods in the `trace.map` call.

The following methods are implemented:
- `AGCAM`: Attention-guided Class Activation Mapping
- `GRADCAM`: Gradient-weighted Class Activation Mapping
- `ATTN`: Raw attention map. Parameters: `sigmoid: bool`.
- `GRAD`: Raw gradient map. Parameters: `relu: bool`, `abs: bool`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
