from collections.abc import Callable, Sequence

import torch
from einops.layers.torch import Rearrange, Reduce
from transformers import PreTrainedModel, ProcessorMixin

from .logger import get_logger
from ..methods import resolve
from .utils import (
    ALL_LAYERS,
    _get_image_token_id,
    _get_vision_patch_shape,
    _select_layers,
)

logger = get_logger(__name__)


class SaliencyTrace:
    """
    Capture activation maps and gradients and compute saliency maps.

    Args:
        model (PreTrainedModel): The model to trace.
        processor (ProcessorMixin): The processor for the model.
        method (str | Callable): The method to use for saliency computation.
        layers (int | object | Sequence[int]): The layers to use for saliency computation.
            - None: use all layers
            - int > 0: use the first `extracted_layers` layers
            - int < 0: use the last `abs(extracted_layers)` layers
            - Sequence[int]: use specific layer indices
        head_reduce (str): The reduction method to use for the head layer.
        layer_reduce (str): The reduction method to use for the intermediate layers.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        *,
        method: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "agcam",
        layers: int | object | Sequence[int] = ALL_LAYERS,
        head_reduce: str = "sum",
        layer_reduce: str = "sum",
    ):
        self.model = model
        self.processor = processor

        # Retrieve image_token_id to identify image vs text tokens.
        self.image_token_id = _get_image_token_id(model.config)
        if self.image_token_id == -1:
            logger.warning(
                "Could not infer image token id from model config. "
                "Please set it manually via `trace.image_token_id = ...`"
            )

        # For models with static vision token counts per image, retrieve it
        self.patch_shape = _get_vision_patch_shape(model.config)
        if self.patch_shape is None:
            logger.info(
                "Image patch shape not found in model config."
                "Falling back to infer it from the input images."
            )

        # Store saliency computation method
        if isinstance(method, str):
            method = resolve(method)
        self.method = method

        # Store saliency aggregation functions
        self.layers = layers
        self.head_reduce = head_reduce
        self.layer_reduce = layer_reduce

        # Runtime caches (populated by capture)
        self._attn: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None

    def capture(
        self,
        *,
        generated_ids: torch.Tensor,  # [1, T_gen]
        input_ids: torch.Tensor,  # [1, T_prompt]
        pixel_values: torch.Tensor,  # [image_count, C, H, W],
        visualize_tokens: bool = False,
        **kwargs: dict | None,
    ):
        """
        Capture activation maps and gradients for saliency computation on a model generation.

        Args:
            generated_ids (torch.Tensor): The token ids generated from the model.
            input_ids (torch.Tensor): The token ids of the input prompt.
            pixel_values (torch.Tensor): The pixel values of the input images.
            visualize_tokens (bool): Whether to create the token visualization widget.
            **kwargs: Additional keyword arguments.

        Throws:
            ValueError: If the input or generated tensors are not 1D.
            ValueError: If no image patch size is set or can be inferred from kwargs[`image_grid_thw`].
            ValueError: If the image tokens shape does not match the expected shape.
        """
        # Ensure batch size is 1
        if (
            generated_ids.ndim != 2
            or input_ids.ndim != 2
            or generated_ids.size(0) != 1
            or input_ids.size(0) != 1
        ):
            raise ValueError("Batch size must be 1 and tensors must be 2D [B,T].")

        # Get image token indices
        image_grid_thw = kwargs.get("image_grid_thw", None)  # Common in Qwen Models
        if image_grid_thw is not None:  # Common in Qwen Models
            image_grid_thw = torch.as_tensor(image_grid_thw).to(
                self.model.device, dtype=torch.int32
            )
            self._image_patch_shapes = (
                image_grid_thw[:, 1:] // 2
            ).tolist()  # each row -> [H, W]
        elif self.patch_shape is None:
            raise ValueError(
                "Image token patch shape not set. Please set `self.patch_shape = (H, W)`"
                "or pass `image_grid_thw` in kwargs (Tensor: [n_images, 3] = [1, H * 2, W * 2])"
            )
        else:
            image_count = pixel_values.shape[0]
            self._image_patch_shapes = [self.patch_shape] * image_count

        # Ensure image sizes line up as expected
        patch_sizes = [H * W for H, W in self._image_patch_shapes]
        expected_img_tkns = sum(patch_sizes)
        image_token_indices = torch.where(input_ids == self.image_token_id)[1]
        if image_token_indices.numel() != expected_img_tkns:
            raise ValueError(
                f"Expected {expected_img_tkns} image tokens, but got {image_token_indices.numel()}"
            )

        splits = torch.split(image_token_indices, patch_sizes)  # Tuple of tensors
        self._image_patches = [t.detach().to(torch.long).cpu() for t in splits]

        device = next(self.model.parameters()).device
        pad_id = self.processor.tokenizer.pad_token_id

        generated_ids = generated_ids.clone().detach().to(device)
        pixel_values = pixel_values.to(device)

        self._gen_start = input_ids.shape[1]
        attention_mask = (generated_ids != pad_id).long().to(device)

        was_training = self.model.training
        self.model.train()

        # Forward pass
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            labels=generated_ids,  # teacher forcing for scalar loss
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

        attn_matrices = list(
            outputs.attentions
        )  # layers * [batch, heads, tokens, tokens]
        for attn in attn_matrices:
            attn.retain_grad()

        # Backward pass
        outputs.loss.backward()

        grad_attn = [attn.grad for attn in attn_matrices]

        # [num_layers, heads, tokens, tokens]
        self._attn = torch.cat([a.detach().cpu() for a in attn_matrices], dim=0)
        self._grad = torch.cat([g.detach().cpu() for g in grad_attn], dim=0)

        self.model.train(was_training)

        if visualize_tokens:
            from ..viz.tokens import render_token_ids

            render_token_ids(
                generated_ids=generated_ids,
                processor=self.processor,
                gen_start=self._gen_start,
                skip_tokens=self.image_token_id,
            )

    def map(
        self,
        token: int,
        *,
        image: int = 0,
        method: str
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        layers: int | object | Sequence[int] | None = None,
        head_reduce: str | None = None,
        layer_reduce: str | None = None,
        **method_kwargs,
    ):
        """
        Compute the saliency map for a specific token and image patch.

        Args:
            token (int): The absolute index of the token to compute the saliency map for.
            image (int): The index of the image patch to compute the saliency map for. Defaults to 0.
            method: If set, overrides the method for computing the saliency map.
            layers: If set, overrides the attention layers used for computing the saliency map.
            head_reduce: If set, overrides the aggregation method for the attention heads.
            layer_reduce: If set, overrides the aggregation method for the layers.
            **method_kwargs: Additional keyword arguments for the saliency map method.
        Returns:
            torch.Tensor: The computed saliency map. [1, 1, H, W]

        Raises:
            ValueError: If no generation has been captured.
            ValueError: If the specified method is unknown.
        """
        if self._attn is None or self._grad is None:
            raise ValueError(
                "No generation has been captured. Please run `capture` first."
            )

        H, W = self._image_patch_shapes[image]
        patch_indices = self._image_patches[image]

        # Use only selected layers for saliency computation
        attn = _select_layers(self._attn, layers or self.layers)
        grad = _select_layers(self._grad, layers or self.layers)

        # Retrieve attention/gradient from token to image
        img_attn = attn[:, :, token, patch_indices]  # [num_layers, heads, patch_size]
        img_grad = grad[:, :, token, patch_indices]  # [num_layers, heads, patch_size]

        # Compute saliency
        if isinstance(method, str):
            method = resolve(method)
        elif method is None:
            method = self.method
        mask = method(img_attn, img_grad, **method_kwargs)

        # Aggregate over heads and layers -> [1, 1, h, w]
        mask = Reduce("l h p -> l p", reduction=head_reduce or self.head_reduce)(mask)
        mask = Reduce("l p -> p", reduction=layer_reduce or self.layer_reduce)(mask)
        mask = Rearrange("(h w) -> 1 1 h w", h=H, w=W)(mask)

        return mask
