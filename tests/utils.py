from dataclasses import dataclass

from transformers.utils.generic import ModelOutput


@dataclass
class DummyOutput(ModelOutput):
    dummy: str


@dataclass
class ImageSpec:
    start: int
    size: tuple[int, int]
