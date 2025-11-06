from collections.abc import Iterable, Iterator
from typing import Protocol

import torch


class Outputs(Protocol):
    """
    A protocol for model output objects.
    Compatible with Hugging Face's ModelOutput.
    """

    loss: torch.Tensor
    attentions: Iterable[torch.Tensor]


class VLModel(Protocol):
    """
    A protocol for VL model objects.
    Compatible with Hugging Face's PreTrainedModel.
    """

    config: object
    training: bool

    def parameters(self, *args, **kwargs) -> Iterator[torch.Tensor]: ...

    def train(self, mode: bool = True) -> None: ...

    def generate(self, *args, **kwargs) -> object: ...

    def zero_grad(self, set_to_none: bool = False) -> None: ...

    def __call__(self, *args, **kwargs) -> Outputs: ...


class Tokenizer(Protocol):
    """
    A protocol for tokenizer objects with a convert_ids_to_tokens method.
    Compatible with Hugging Face's PreTrainedTokenizer.
    """

    def convert_ids_to_tokens(
        self,
        ids: list[int],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]: ...


class Processor(Protocol):
    """
    A protocol for processor objects with a tokenizer attribute.
    Compatible with Hugging Face's ProcessorMixin.
    """

    @property
    def tokenizer(self) -> Tokenizer: ...
