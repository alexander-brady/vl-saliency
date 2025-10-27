from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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


class Chainable:
    """Mixin that provides `>>` composition."""

    def __rshift__(self, other: Chainable) -> Pipeline:
        if isinstance(other, Pipeline):
            return Pipeline(self, *other.transforms)
        return Pipeline(self, other)

    def __call__(self, map: SaliencyMap) -> SaliencyMap:
        raise NotImplementedError("Chainable subclasses must implement __call__.")


def chainable(fn: Transform) -> Callable[..., Pipeline | SaliencyMap]:
    """Decorator to make any function or method chainable with >>."""

    @wraps(fn)
    def wrapper(*args, **kwargs) -> Pipeline | SaliencyMap:
        from ..core.map import SaliencyMap

        # EAGER: fn(smap, ...) if first arg is SaliencyMap
        if args and isinstance(args[0], SaliencyMap):
            return fn(*args, **kwargs)

        # LAZY: fn(...) returns Pipeline otherwise
        def transform(map: SaliencyMap) -> SaliencyMap:
            return fn(map, *args, **kwargs)

        return Pipeline(transform)

    return wrapper


class Pipeline(Chainable):
    """
    A pipeline of multiple transforms to be applied sequentially to a SaliencyMap.

    Attributes:
        transforms (list[Transform]): A list of Transform instances to be applied in sequence.
    """

    def __init__(self, *transforms: Transform):
        self.transforms = list(transforms)

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
