"""Saliency visualizer, modified from
https://github.com/LeemSaebom/Attention-Guided-CAM-Visual-Explanations-of-Vision-Transformer-Guided-by-Self-Attention/
"""
from typing import Tuple, Sequence, Optional
import torch
import torch.nn.functional as F
from einops.layers.torch import Reduce, Rearrange
from transformers import PreTrainedModel, ProcessorMixin

_KEEP = object()

class AGCAM:
    """
    Implementation of Attention-Guided CAM (AGCAM) for Vision Transformers.
    
    Arguments:
        model (PreTrainedModel): the Vision Transformer model to be explained
        processor (ProcessorMixin): the processor for the model's input        
        pad_token_id (int): the token ID used for padding in the model's vocabulary
        image_token_id (int): the token ID used for image patches in the model's vocabulary (default: 262144 for Gemma models)
        image_patch_size (Union[int, Tuple[int, int]]): size of the image patches (default: (16, 16))
        head_fusion (str): type of head-wise aggregation (default: 'sum')
        layer_fusion (str): type of layer-wise aggregation (default: 'sum') 
        extracted_layers (None | int | Sequence[int]):        
                - None: use all layers
                - int > 0: use the first `extracted_layers` layers
                - int < 0: use the last `abs(extracted_layers)` layers
                - Sequence[int]: use specific layer indices
    """
    def __init__(
        self, 
        model: PreTrainedModel,
        processor: ProcessorMixin,
        image_token_id: int = 262144,  # Image soft token ID for Gemma models
        image_patch_size: int | Tuple[int, int] = (16, 16),
        head_fusion: str = 'sum', 
        layer_fusion: str = 'sum',
        extracted_layers: int | Sequence[int] | None = None
    ):
        self.model = model
        self.processor = processor
        self.image_token_id = image_token_id
        
        if isinstance(image_patch_size, int):
            self.height = image_patch_size
            self.width = image_patch_size
        else:
            self.height, self.width = image_patch_size
            
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion
        self.extracted_layers = extracted_layers
        
        self._attn = None
        self._grad = None
        self._generated_ids = None
        self._image_patches = None

    def generate(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate model response and prepare the attention and gradients for saliency computation.
        
        Args:
            inputs (dict): processed inputs to the model, typically including 'input_ids', 'attention_mask', etc.
            
        Returns:
            generated_ids: the generated token ids from the model, shape [1, sequence_length]
        """
        # Forward pass with generation
        with torch.no_grad():
            generated = self.model.generate(**inputs, return_dict_in_generate=True)
        generated_ids = generated.sequences

        input_ids = generated_ids.clone().detach()
        input_ids.requires_grad = False # do not compute gradients for input_ids, only attention

        # Build attention mask
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long()

        self.model.zero_grad()
        
        # Forward pass with labels to compute loss and capture attention
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # teacher forcing for loss computation
            output_attentions=True, 
            return_dict=True,
        )
        
        attn_matrices = outputs.attentions  # List of [batch, heads, tokens, tokens]
        
        for attn in outputs.attentions:
            attn.retain_grad()
            
        # Compute loss and backpropagate to get gradients w.r.t. attention
        loss = outputs.loss
        loss.backward()
        
        grad_attn = [a.grad for a in attn_matrices]
        
        self._attn = attn = torch.cat(attn_matrices, dim=0) # [num_layers, batch, heads, tokens, tokens]
        self._grad = torch.cat(grad_attn, dim=0)            # [num_layers, batch, heads, tokens, tokens]
        self._generated_ids = generated_ids
                
        # extract token indices where token is self.image_token_id
        image_token_indices = torch.where(input_ids[0] == self.image_token_id)[0]
        num_patches = self.height * self.width

        assert image_token_indices.numel() % num_patches == 0, \
            f"Number of image tokens should be `image count * height ({self.height}) * width ({self.width})`, but got `{len(image_token_indices)}`"

        # Reshape to [num_images, height * width]
        self._image_patches = image_token_indices.view(-1, num_patches)  # [num_images, num_patches]
        
        return generated_ids    
    
    def compute_saliency(
        self, 
        token_id: int, 
        image_id: int = 0, 
        *,
        head_fusion: Optional[str] = None,
        layer_fusion: Optional[str] = None,
        extracted_layers: None | int | Sequence[int] | object = _KEEP,
    ) -> torch.Tensor:
        """
        Compute the saliency map for a specific token in the image. 
        
        Args:
            token_id (int): The token ID to compute the saliency for.
            image_id (int): The index of the image to compute the saliency for (default: 0).
            extracted_layers (None | int | Sequence[int]): Override the layers to extract for this saliency computation.
            head_fusion (Optional[str]): Override the head-wise aggregation for this saliency computation.
            layer_fusion (Optional[str]): Override the layer-wise aggregation for this saliency computation.
            
        Returns:
            torch.Tensor: The computed saliency map for the specified token and image.
            
        Raises:
            ValueError: If the attention or gradients have not been computed yet.
        """
        if self._attn is None or self._grad is None:
            raise ValueError("You must call `generate` first to compute attention and gradients.")
        
        def extract_layers(data: torch.Tensor, indices: None | int | Sequence[int] = None) -> torch.Tensor:
            if indices is None:
                return data
            if isinstance(indices, int):
                return data[indices:] if indices > 0 else data[:indices]
            return torch.stack([data[i] for i in indices if i < len(data)])

        extracted_layers = extracted_layers if extracted_layers is not _KEEP else self.extracted_layers
        attn = extract_layers(self._attn, extracted_layers)
        grad = extract_layers(self._grad, extracted_layers)

        patch_indices = self._image_patches[image_id]

        attn_img = attn[:, :, token_id, patch_indices]  # [layers, heads, patch_tokens]
        grad_img = grad[:, :, token_id, patch_indices]  # [layers, heads, patch_tokens]

        grad_img = F.relu(grad_img)
        attn_img = torch.sigmoid(attn_img)
        mask = grad_img * attn_img
        
        head_fusion = head_fusion or self.head_fusion
        layer_fusion = layer_fusion or self.layer_fusion

        mask = Reduce('l h p -> l p', reduction=head_fusion)(mask)
        mask = Reduce('l p -> p', reduction=layer_fusion)(mask)
        mask = Rearrange('(h w) -> h w', h=self.height, w=self.width)(mask)

        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
        return mask