from __future__ import annotations

from typing import NamedTuple

type IndexLike = int | tuple[int] | tuple[int, int] | tuple[int, int, int] | Index


class Index(NamedTuple):
    """Structured index for accessing saliency maps."""

    batch_idx: int | None = None
    """Which batch item to access. None only if there is a single batch item."""
    img_idx: int | None = None
    """Which image to access. None only if there is a single image in all batch items."""
    token_idx: int | None = None
    """Which token to access."""

    @classmethod
    def from_indices(cls, idx: IndexLike) -> Index:
        """Converts index formats into a structured Index object."""
        if isinstance(idx, cls):
            return idx

        if isinstance(idx, int):
            return cls(token_idx=idx)

        if isinstance(idx, tuple):
            padded = (None,) * (3 - len(idx)) + idx
            return cls(*padded)
