from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Protocol, cast, overload, runtime_checkable

if TYPE_CHECKING:
    from ..core.map import SaliencyMap


@runtime_checkable
class Transform(Protocol):
    """A transformation that can be applied to a SaliencyMap."""

    def __call__(self, map: SaliencyMap) -> SaliencyMap: ...


@runtime_checkable
class TraceTransform(Protocol):
    """A transformation that requires both attention and gradient data from a Trace."""

    def __call__(self, attn: SaliencyMap, grad: SaliencyMap) -> SaliencyMap: ...


class ChainableCallable(Protocol):
    """Defines a callable that can be chained with >> to form a Pipeline or directly called."""

    @overload
    def __call__(self, map: SaliencyMap, /) -> SaliencyMap: ...
    @overload
    def __call__(self) -> Pipeline: ...


class Chainable(ABC):
    """Mixin that provides `>>` composition."""

    def __rshift__(self, other: Chainable) -> Pipeline:
        if isinstance(other, Pipeline):
            return Pipeline(self, *other.transforms)
        return Pipeline(self, other)

    @abstractmethod
    def __call__(self, map: SaliencyMap) -> SaliencyMap: ...  # To be implemented by subclasses


def chainable(fn: Transform) -> ChainableCallable:
    """Decorator to make any function or method chainable with >>."""

    @wraps(fn)
    def wrapper(*args) -> Pipeline | SaliencyMap:
        from ..core.map import SaliencyMap

        # EAGER: fn(saliency_map) called directly
        if args and isinstance(args[0], SaliencyMap):
            return fn(map=args[0])

        # LAZY: fn return Pipeline otherwise
        def transform(map: SaliencyMap) -> SaliencyMap:
            return fn(map=map)

        return Pipeline(transform)

    return cast(ChainableCallable, wrapper)


class Pipeline(Chainable):
    """
    A pipeline of multiple transforms to be applied sequentially to a SaliencyMap.

    Attributes:
        transforms (list[Transform]): A list of Transform instances to be applied in sequence.
    """

    def __init__(self, *transforms: Transform):
        self.transforms = list(transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __rshift__(self, other: Transform) -> Pipeline:
        if isinstance(other, Pipeline):
            return Pipeline(*self.transforms, *other.transforms)
        return Pipeline(*self.transforms, other)

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        for transform in self.transforms:
            map = map.apply(transform)
        return map

    def __repr__(self) -> str:
        transform_names = [type(t).__name__ for t in self.transforms]
        return f"Pipeline({', '.join(transform_names)})"

    def append(self, other: Transform) -> None:
        """Append a transform or another pipeline to this pipeline."""
        if isinstance(other, Pipeline):
            self.transforms.extend(other.transforms)
        else:
            self.transforms.append(other)
