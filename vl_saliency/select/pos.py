from vl_saliency.core.scoped import ScopedSaliencyGrid


class AbsoluteIndex:
    """
    Selects token index based on an absolute index (as opposed to relative to the generated tokens).

    Attributes:
        index (int): The absolute index of the token to select.

    Raises:
        IndexError: If the index is out of bounds for the given scoped saliency grid.
    """

    def __init__(self, index: int):
        self.index = index

    def __call__(self, scoped: ScopedSaliencyGrid) -> int:
        if not scoped.gen_start_idx <= self.index < scoped.gen_end_idx:
            raise IndexError(
                f"Index {self.index} is out of bounds for the generated tokens "
                f"in batch {scoped.batch_idx}, image {scoped.image_idx}. "
                f"Valid range: [{scoped.gen_start_idx}, {scoped.gen_end_idx})"
            )

        return self.index - scoped.gen_start_idx


class ReverseIndex:
    """
    Selects a token based on a reverse index (counting from the end).

    Attributes:
        index (int): The reverse index of the token to select (0-based from the end).
    Raises:
        IndexError: If the reverse index is out of bounds for the given scoped saliency grid.
    """

    def __init__(self, offset_from_end: int):
        if offset_from_end < 0:
            raise IndexError(f"Reverse index must be non-negative, got {offset_from_end}.")
        self.offset_from_end = offset_from_end

    def __call__(self, scoped: ScopedSaliencyGrid) -> int:
        if not 0 <= self.offset_from_end < scoped.num_tokens:
            raise IndexError(
                f"Reverse index {self.offset_from_end} is out of bounds for the generated tokens "
                f"in batch {scoped.batch_idx}, image {scoped.image_idx}. "
                f"Valid range: [0, {scoped.num_tokens})"
            )

        return scoped.num_tokens - 1 - self.offset_from_end
