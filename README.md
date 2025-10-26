# Vision–Language Saliency Extraction

[![CI](https://github.com/alexander-brady/vl-saliency/actions/workflows/ci.yml/badge.svg)](https://github.com/alexander-brady/vl-saliency/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vl-saliency.svg)](https://pypi.org/project/vl-saliency/)
[![Python](https://img.shields.io/badge/python-≥3.10-purple.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/github/license/alexander-brady/vl-saliency.svg)](https://github.com/alexander-brady/vl-saliency/blob/main/LICENSE)

This library provides a simple, model-agnostic interface to compute and visualize text-to-image saliency maps, extending classic methods originally developed for Vision Transformers (ViTs) to modern vision-language architectures. Compatible with any Hugging Face Image-Text-to-Text model, this library makes it easy to interpret vision-language model output. Modular and extensible, novel saliency techniques can be easily integrated.

**Table of Contents**

- [Installation](#installation)
- [Features](#features)
- [Attention and Gradients](#attention-and-gradients)
- [Transforms](#transforms)
- [Pipeline API](#pipeline-api)
- [Defining Custom Transforms](#defining-custom-transforms)

## Installation

This library is available through PyPI and can be installed using pip:

```bash
pip install vl-saliency
```

## Features

> See the [quickstart notebook](notebooks/quickstart.ipynb) for a complete example of how to use the saliency extractor with a Gemma3 vision-language model.

Using `SaliencyExtractor` objects, you can easily compute and visualize saliency maps for any Hugging Face Image-Text-to-Text model.

```python
from vl_saliency import SaliencyExtractor

# Initialize the model and input prompt
model = AutoModel.from_pretrained("model_name")  # Replace with your model name
processor = AutoProcessor.from_pretrained("model_name")  # Replace with your processor name

image = PIL.Image.open("path_to_image.jpg")  # Load your image
inputs = processor(text="Your prompt", images=image, return_tensors="pt")

# Initialize the saliency extractor
extractor = SaliencyExtractor(model, processor)

# Generate response 
with torch.inference_mode():
    generated_ids = model.generate(**inputs, do_sample=True, max_new_tokens=200) 
    
# Compute attention and gradients
trace = extractor.capture(**inputs, generated_ids=generated_ids)

# Compute the saliency map from a specific token to the image
saliency_map = trace.map(token=200)  # Change token_index as needed

# Aggregate the saliency map's layers and heads
saliency_map = saliency_map.agg(layer_reduce="mean", head_reduce="mean")

# Visualize the saliency map
saliency_map.plot(image, title="Saliency Map")
```

## Attention and Gradients

You can compute saliency maps based on either attention weights or gradients. By default, `SaliencyExtractor` stores both attention and gradient information during the forward and backward passes. If you only need one of these, you can disable the other to save memory and computation time.

```python
# Initialize the saliency extractor to store only gradients
extractor = SaliencyExtractor(model, processor, store_attns=False) # Similarly, use store_grads=False to store only attention

saliency_map = extractor.capture(**inputs, generated_ids=generated_ids).map(token=200)
saliency_map.agg().plot(image, title="Gradient-based Saliency Map")
```

Some more advanced saliency methods may require access to both attention weights and gradients. You can apply such methods directly to traces using the `>>` operator or the `apply` method, returning a new saliency map.

```python
from vl_saliency.lib import gradcam

# Compute Grad-CAM saliency map
saliency_map = trace.apply(token=200, transform=gradcam)
saliency_map.agg().plot(image, title="Grad-CAM Saliency Map")
```

To define your own such composite saliency methods, see the [Defining Custom Transforms](#defining-custom-transforms) section below.


## Transforms

The library includes several built-in `Transform` objects to process saliency maps. Saliency maps are immutable, so applying a transform returns a new saliency map. You can chain transforms using the `>>` operator, or call the `apply` method.

```python
from vl_saliency import transforms as T

# Example: Normalize and plot a saliency map
saliency_map = saliency_map >> T.normalize()
saliency_map.agg().plot(image, title="Normalized Saliency Map")

# Example: Binarize a saliency map, setting values below the mean to zero
saliency_map = saliency_map.apply(T.Binarize(threshold="mean"))
saliency_map.agg().plot(image, title="Binarized Saliency Map")

# Example: Apply the sigmoid function to a saliency map, then aggregate across heads and layers
saliency_map = saliency_map >> T.Sigmoid() >> T.Aggregate(layer_reduce="mean", head_reduce="mean")
saliency_map.plot(image, title="Sigmoid Saliency Map")
```

## Pipeline API

For more complex visualization workflows, you can combine multiple `Transform` objects into a reuseable `Pipeline`, allowing you to apply the same sequence of transforms to multiple saliency maps.

```python
from vl_saliency import transforms as T

pipe = (
    T.abs() >>
    T.normalize() >>
    T.Aggregate(layer_reduce="mean", head_reduce="sum")
)

# Apply the pipeline to a saliency map
saliency_map >>= pipe
saliency_map.plot(image, title="Pipeline Processed Saliency Map")

# Alternatively, you can directly create a pipeline using the constructor
pipe = T.Pipeline(
    T.abs(),
    T.normalize(),
    T.Aggregate(layer_reduce="mean", head_reduce="sum")
)

saliency_map = saliency_map.apply(pipe).plot(image, title="Pipeline Processed Saliency Map")
```

## Defining Custom Transforms

You can define your own custom transforms by subclassing the `Chainable` interface. Note that `Chainable` classes must implement the `__call__` method with exactly the following signature:

```python
from vl_saliency import SaliencyMap
from vl_saliency.transforms import Chainable

class MyTransform(Chainable):
    def __call__(self, saliency_map: SaliencyMap) -> SaliencyMap:
        # Custom transformation logic
        return saliency_map
```

Alternatively, you can use the `@chainable` decorator to create simple transforms without subclassing. The decorated function must also adhere to the same signature:

```python
from vl_saliency import SaliencyMap
from vl_saliency.transforms import chainable

@chainable
def my_custom_transform(saliency_map: SaliencyMap) -> SaliencyMap:
    # Custom transformation logic
    return saliency_map
```

For methods that require both attention weights and gradients, you can define a transform that processes both and returns a new saliency map. Such transforms are defined under the protocol `TraceTransform` and can be applied directly to `Trace` objects using the `apply` method. They must implement the following signature:

```python
from vl_saliency import SaliencyMap

def my_trace_transform(attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
    # Custom transformation logic using both attention and gradients
    return saliency_map

class MyTraceTransform:
    def __call__(self, attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap:
        # Custom transformation logic using both attention and gradients
        return saliency_map
```

Note that `TraceTransform` objects aren't chainable like regular transforms, since they operate on two inputs.

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
