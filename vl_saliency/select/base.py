from __future__ import annotations

from abc import ABC, abstractmethod

from vl_saliency.maps.view import SaliencyView


class IndexSelector(ABC):
    """Selects token indices for a ScopedSaliencyGrid,
    enabling dynamic access patterns based on token content or position."""

    def __call__(self, view: SaliencyView) -> int:
        idx = self.select(view)

        if not 0 <= idx < view.num_tokens:
            raise IndexError(
                f"Token index {idx} out of bounds for {view.num_tokens} tokens "
                f"(batch={view.batch_idx}, image={view.image_idx})"
            )

        return idx

    @abstractmethod
    def select(self, view: SaliencyView) -> int: ...

    def __add__(self, offset: int) -> IndexSelector:
        return OffsetSelector(self, offset)

    def __sub__(self, offset: int) -> IndexSelector:
        return OffsetSelector(self, -offset)


class OffsetSelector(IndexSelector):
    def __init__(self, selector: IndexSelector, offset: int):
        self.selector = selector
        self.offset = offset

    def select(self, view: SaliencyView) -> int:
        return self.selector(view) + self.offset
