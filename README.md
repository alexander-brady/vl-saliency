# Vision–Language Saliency Extraction

[![CI](https://github.com/alexander-brady/vl-saliency/actions/workflows/ci.yml/badge.svg)](https://github.com/alexander-brady/vl-saliency/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vl-saliency.svg)](https://pypi.org/project/vl-saliency/)
[![Python](https://img.shields.io/badge/python-≥3.11-purple.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/github/license/alexander-brady/vl-saliency.svg)](https://github.com/alexander-brady/vl-saliency/blob/main/LICENSE)

This library provides a simple, model-agnostic interface to compute and visualize text-to-image saliency maps, extending classic methods originally developed for Vision Transformers (ViTs) to modern vision-language architectures. Compatible with any Hugging Face Image-Text-to-Text model, this library makes it easy to interpret vision-language model output. Modular and extensible, novel saliency techniques can be easily integrated.

**Table of Contents**

- [Installation](#installation)
- [Features](#features)
- [Attention and Gradients](#attention-and-gradients)

## Installation

This library is available through PyPI and can be installed using pip:

```bash
pip install vl-saliency
```

## Features

> See the [quickstart notebook](notebooks/quickstart.ipynb) for a complete example of how to use the saliency extractor with a Gemma3 vision-language model.

Using `SaliencyExtractor` objects, you can easily compute and visualize saliency maps for any Hugging Face Image-Text-to-Text model.

> **Limitation**: This library currently only supports inputs with exactly one image per item in the batch (or completely image-free batches). Support for multiple images per item will be added in a future release.

```python
from vl_saliency import Saliency
from vl_saliency.viz import plot_saliency_map

# Initialize the model and input prompt
model = AutoModel.from_pretrained("model_name")  # Replace with your model name
processor = AutoProcessor.from_pretrained("model_name")  # Replace with your processor name

image = PIL.Image.open("path_to_image.jpg")  # Load your image
inputs = processor(text="Your prompt", images=image, return_tensors="pt")

# Generate response and saliency maps in a single step
with Saliency(model):
    output = model(**inputs)

sal = output.saliency

# Compute a specific saliency map from a specific token to the image
saliency_map = sal.map(token_idx=200)  # Change token index as needed

# Visualize the saliency map
plot_saliency_map(saliency_map, image, title="Saliency Map")
```

## Attention and Gradients

You can compute saliency maps based on the model's attention weights. Alternatively, you can compute gradient-based saliency maps by back-propagating from the generated token of interest to the image tokens. 

```python
extractor = SaliencyExtractor(model)
extractor.wrap() # Equivalent to using the context manager

output = model(**inputs)
sal = output.saliency

output.loss.backward()  # Backpropagate to compute gradients

saliency_map = sal.map(token_idx=200)  # Attention-based saliency map

# saliency_map is a tensor, so we can retrieve the gradients directly
grad_map = saliency_map.grad 
```

## Contributing

Contributions are welcome! Open an issue to discuss ideas or submit a PR directly.

**Getting Started**

1. Clone the repository and install the required dependencies.

    ```bash
    git clone https://github.com/alexander-brady/vl-saliency
    cd vl-saliency
    ```

2. Create a virtual environment and activate it.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the development dependencies.
    ```bash
    pip install -e .[dev]
    ```

**Guidelines**

Before submitting a pull request, ensure:
```
ruff check . --fix && ruff format .   # Lint & format
pytest                                # Run tests
mypy .                                # Type check
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
