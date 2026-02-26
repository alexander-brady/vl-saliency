from dataclasses import dataclass, is_dataclass

from transformers.utils.generic import ModelOutput

from vl_saliency.core.grid import SaliencyGrid


@dataclass
class SaliencyOutput(ModelOutput):
    """Output wrapper for model outputs that includes the saliency map alongside the original model output.
    Fields from the original model output can be accessed directly on this object,
    and the saliency map is available as the `saliency` attribute."""

    saliency: SaliencyGrid
    """Saliency map computed during the forward pass."""
    base_output: ModelOutput | None = None
    """The original output from the model's forward pass."""

    def __getattr__(self, name: str):
        if is_dataclass(self.base_output) and name in self.base_output.__dataclass_fields__:
            return getattr(self.base_output, name)
        return super().__getattribute__(name)
