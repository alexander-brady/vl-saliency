import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin

from ..utils.logger import get_logger
from ..utils.transformer_utils import _get_image_token_id, _get_vision_patch_shape, _image_patch_shapes
from .trace import Trace

logger = get_logger(__name__)


class SaliencyExtractor:
    """
    Engine to capture attention and gradient data from a vision-language model during inference.

    Attributes:
        model (PreTrainedModel): The vision-language model to trace.
        processor (ProcessorMixin): The processor for tokenization and decoding.
        store_grads (bool, default=True): Whether to store gradients during tracing.
        store_attns (bool, default=True): Whether to store attention weights during tracing.

    Methods:
        capture(generated_ids, **inputs, store_grads): Capture a Trace from a model inference with processed inputs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        store_attns: bool = True,
        store_grads: bool = True,
    ):
        self.model = model
        self.processor = processor
        self.store_grads = store_grads
        self.store_attns = store_attns

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
            logger.info("Image patch shape not found in model config. Falling back to infer it from the input images.")

    def capture(
        self,
        generated_ids: torch.Tensor,  # [1, T_gen]
        *,
        store_grads: bool | None = None,
        store_attns: bool | None = None,
        input_ids: torch.Tensor,  # [1, T_prompt]
        pixel_values: torch.Tensor,  # [image_count, C, H, W],
        image_grid_thw: torch.Tensor | None = None,  # [image_count, 3]
    ) -> Trace:
        """
        Capture a Trace from a model inference with given inputs.

        Recommended to use processed inputs from the processor for best results.

        Example:
        ```python
            inputs = processor(images, text=prompt, return_tensors="pt")
            generated_ids = model.generate(**inputs)
            trace = engine.capture(generated_ids, **inputs, store_grads=True)
        ```

        Args:
            generated_ids (torch.Tensor): Tensor of generated token IDs during inference. Shape: [1, T_gen].
            store_grads (bool | None, default=None): Whether to store gradients during tracing. If None, uses the engine's default.
            store_attns (bool | None, default=None): Whether to store attention weights during tracing. If None, uses the engine's default.
            input_ids (torch.Tensor): Tensor of input token IDs (prompt). Shape: [1, T_prompt].
            pixel_values (torch.Tensor): Tensor of input images. Shape: [image_count, C, H, W].
            image_grid_thw (torch.Tensor | None, default=None): Optional tensor specifying the grid size (thw) for each image. Shape: [image_count, 3]. Common in Qwen models.

        Returns:
            Trace: The captured Trace containing attention and gradient data.

        Raises:
            ValueError: If input dimensions are incorrect or if image token counts do not match expectations.
        """
        # Ensure at least one of attention or gradients is stored
        store_attns = store_attns if store_attns is not None else self.store_attns
        store_grads = store_grads if store_grads is not None else self.store_grads

        if not store_attns and not store_grads:
            raise ValueError("At least one of store_attns or store_grads must be True to capture a trace.")

        # Ensure batch size is 1
        if generated_ids.ndim != 2 or input_ids.ndim != 2 or generated_ids.size(0) != 1 or input_ids.size(0) != 1:
            raise ValueError("Batch size must be 1 and tensors must be 2D [B,T].")

        # Ensure generated_ids is an extension of input_ids
        if generated_ids.size(1) < input_ids.size(1) or not torch.equal(
            generated_ids[:, : input_ids.size(1)], input_ids
        ):
            raise ValueError("generated_ids must extend input_ids.")

        # Ensure at least one image is provided
        image_count = pixel_values.shape[0]
        if image_count < 1:
            raise ValueError("At least one image must be provided in pixel_values.")

        # Get image token indices
        patch_shapes = _image_patch_shapes(
            image_count=image_count, patch_shape=self.patch_shape, image_grid_thw=image_grid_thw
        )

        # Ensure image sizes line up as expected
        patch_sizes = [H * W for H, W in patch_shapes]
        expected_image_tokens = sum(patch_sizes)
        image_token_indices = torch.where(input_ids == self.image_token_id)[1]
        if image_token_indices.numel() != expected_image_tokens:
            raise ValueError(
                f"Number of image tokens in input_ids ({image_token_indices.numel()}) does not match expected "
                f"count from image sizes ({expected_image_tokens}). Please check `image_token_id` and input images."
            )

        # Compute individual image patches
        splits = torch.split(image_token_indices, patch_sizes)
        image_patches = [t.detach().to(torch.long).cpu() for t in splits]

        # Determine whether to store gradients
        if store_grads is None:
            store_grads = self.store_grads

        device = next(self.model.parameters()).device
        pad_id = self.processor.tokenizer.pad_token_id  # type: ignore

        generated_ids = generated_ids.clone().detach().to(device)
        pixel_values = pixel_values.to(device)

        gen_start = input_ids.shape[1]
        attention_mask = (generated_ids != pad_id).long().to(device)

        was_training = self.model.training
        self.model.train(store_grads)  # Enable gradients if needed

        labels = generated_ids if store_grads else None
        context = torch.enable_grad() if store_grads else torch.no_grad()

        # Forward pass
        with context:
            if store_grads:
                self.model.zero_grad(set_to_none=True)

            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                labels=labels,  # teacher forcing for scalar loss
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )

        attn_matrices = list(outputs.attentions)  # layers * [batch, heads, tokens, tokens]

        attn = None
        grad = None

        # Backward pass
        if store_grads:
            for a in attn_matrices:
                a.retain_grad()

            outputs.loss.backward()

            grad = torch.cat(
                [a.grad.detach().cpu() for a in attn_matrices], dim=0
            )  # [num_layers, heads, tokens, tokens]
            grad = grad[:, :, gen_start:, :]  # Keep only generated tokens

        if store_attns:
            attn = torch.cat([a.detach().cpu() for a in attn_matrices], dim=0)  # [num_layers, heads, tokens, tokens]
            attn = attn[:, :, gen_start:, :]  # Keep only generated tokens

        # Restore model training state
        self.model.train(was_training)

        # Keep only the text-to-image attention/gradients
        text2img_attn: list[torch.Tensor] | None = [] if attn is not None else None
        text2img_grad: list[torch.Tensor] | None = [] if grad is not None else None

        # Helper to collect text-to-image attentions/gradients
        def collect_text2img(indices, tensor, H, W, collect_list):
            if tensor is not None and collect_list is not None:
                t = tensor.index_select(dim=-1, index=indices)  # [layers, heads, gen_tokens, image_tokens]
                t = t.contiguous().view(*t.shape[:-1], H, W)  # [layers, heads, gen_tokens, H, W]
                collect_list.append(t)

        # Populate text-to-image lists
        for indices, (H, W) in zip(image_patches, patch_shapes, strict=True):
            collect_text2img(indices, attn, H, W, text2img_attn)
            collect_text2img(indices, grad, H, W, text2img_grad)

        # Construct and return Trace
        return Trace(
            attn=text2img_attn,
            grad=text2img_grad,
            processor=self.processor,
            image_token_id=self.image_token_id,
            gen_start=gen_start,
            generated_ids=generated_ids.detach().cpu(),
        )
