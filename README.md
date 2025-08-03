# Vision-Language Saliency Extraction

This repository provides tools for extracting saliency maps from vision-language models, extending the [Attention-Guided CAM (AGCAM)](https://github.com/LeemSaebom/Attention-Guided-CAM-Visual-Explanations-of-Vision-Transformer-Guided-by-Self-Attention) method originally developed for Vision Transformers (ViTs) to vision-language architectures.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
2. Install the required packages:
   ```bash
    pip install -e . # Install the package in editable mode
    pip install -r requirements.txt
    ```
    
## Usage

> See the [example notebook](notebooks/gemma.ipynb) for a complete example of how to use the AGCAM saliency extractor with a Gemma3 vision-language model.

To extract saliency maps for a vision-language model using AGCAM, you can use the following code snippet:

```python
from vl_saliency import AGCAM
from vl_saliency.utils import visualize_saliency

# Initialize the model and input prompt
model = AutoModel.from_pretrained("model_name")  # Replace with your model name
processor = AutoProcessor.from_pretrained("model_name")  # Replace with your processor name

image = PIL.Image.open("path_to_image.jpg")  # Load your image
inputs = processor(text="Your input text", images=image, return_tensors="pt")

# Initialize the AGCAM saliency extractor
visualizer = AGCAM(
  model=model,
  pad_token_id=processor.tokenizer.pad_token_id,
  image_token_id=262144,  # Image soft token ID for your models
  image_patch_size=16,  # Patch size for the your model
)

# Generate response and compute saliency map
response = visualizer.generate(**inputs)
saliency_map = visualizer.compute_saliency(response, token_index=0)  # Change token_index as needed

# Visualize the saliency map
fig = visualize_saliency(saliency_map, image, colormap='viridis', title="Saliency Map")
fig.show()  # or save the plot using fig.savefig("saliency_map.png")
```

## Attribution

We are not affiliated with the authors of the AGCAM paper. If you use this code, please cite the original authors:

```bibtex
@inproceedings{leem2024attention,
  title={Attention guided CAM: visual explanations of vision transformer guided by self-attention},
  author={Leem, Saebom and Seo, Hyunseok},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={4},
  pages={2956--2964},
  year={2024}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.