"""Saliency visualizer, modified from
https://github.com/LeemSaebom/Attention-Guided-CAM-Visual-Explanations-of-Vision-Transformer-Guided-by-Self-Attention/
"""
from typing import Tuple, Union
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from einops.layers.torch import Reduce, Rearrange

class AGCAM:
    """
    Implementation of Attention-Guided CAM (AGCAM) for Vision Transformers.
    
    Arguments:
        model (PreTrainedModel): the Vision Transformer model to be explained
        pad_token_id (int): the token ID used for padding in the model's vocabulary
        image_token_id (int): the token ID used for image patches in the model's vocabulary (default: 262144 for Gemma models)
        image_patch_size (Union[int, Tuple[int, int]]): size of the image patches (default: (16, 16))
        head_fusion (str): type of head-wise aggregation (default: 'sum')
        layer_fusion (str): type of layer-wise aggregation (default: 'sum')    
    """
    def __init__(
        self, 
        model: PreTrainedModel,
        pad_token_id: int,
        image_token_id: int = 262144,  # Image soft token ID for Gemma models
        image_patch_size: Union[int, Tuple[int, int]] = (16, 16),
        head_fusion: str = 'sum', 
        layer_fusion: str = 'sum'
    ):
        self.model = model
        self.image_token_id = image_token_id
        
        if isinstance(image_patch_size, int):
            self.height = image_patch_size
            self.width = image_patch_size
        else:
            self.height, self.width = image_patch_size
            
        self.pad_token_id = pad_token_id
        self.head_fusion = head_fusion
        self.layer_fusion = layer_fusion


    def generate(self, **inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Generate the attention-guided CAM for the given inputs.
        TODO: Update this to support multiple images (returning a list of tuples)
        
        Args:
            inputs (dict): inputs to the model, typically including 'input_ids', 'attention_mask', etc.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - generated_ids: the generated token ids from the model
                - mask: the attention-guided CAM mask, shape [1, 1, height, width]
        """        
        # Forward pass with generation
        with torch.no_grad():  # allows forward pass without locking tensors
            generated = self.model.generate(**inputs, return_dict_in_generate=True)
        generated_ids = generated.sequences

        # Forward + backward pass to compute gradients for those tokens
        input_ids = generated_ids.clone().detach()
        input_ids.requires_grad = False  # no need to track generated_ids

        # Build attention mask
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # Backpropagate the model from the prediction
        self.model.zero_grad()
        
        # To compute gradients, we need to do a forward pass with labels
        # Shift inputs appropriately depending on whether it's encoder-decoder or causal LM
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # forces teacher forcing loss computation
            output_attentions=True,  # to get attention matrices
            return_dict=True,
        )
        
        attn_matrices = outputs.attentions  # List of length = num_layers   
        # Each element: [batch, heads, tokens, tokens]
        for attn in outputs.attentions:
            attn.retain_grad()
    
        
        # Compute loss and backpropagate
        loss = outputs.loss
        loss.backward()
        
        grad_attn = [a.grad for a in attn_matrices]

        # Concatenate attention matrices and gradients
        attn = attn_matrices[0]
        grad = grad_attn[0]
        for i in range(1, len(attn_matrices)):
            attn = torch.cat((attn, attn_matrices[i]), dim=0)
            grad = torch.cat((grad, grad_attn[i]), dim=0)
        
        # extract token indices where token is self.image_token_id
        image_token_indices = torch.where(input_ids[0] == self.image_token_id)[0]
        assert len(image_token_indices) == self.height * self.width, \
            f"Expected {self.height * self.width} image tokens, got {len(image_token_indices)}"
                
        attn = attn[:, :, 0, image_token_indices]     # [layers, heads, patch_tokens]
        grad = grad[:, :, 0, image_token_indices]     # [layers, heads, patch_tokens]

        # Saliency weighting
        grad = F.relu(grad)
        attn = torch.sigmoid(attn)
        mask = grad * attn
        
        mask = Reduce('l h p -> l p', reduction=self.head_fusion)(mask)
        mask = Reduce('l p -> p', reduction=self.layer_fusion)(mask)
        mask = Rearrange('(h w) -> h w', h=self.height, w=self.width)(mask)

        # Add batch and channel dimensions for compatibility
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
        
        return generated_ids, mask