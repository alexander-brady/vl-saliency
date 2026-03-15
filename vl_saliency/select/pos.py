from vl_saliency.maps.view import SaliencyView
from vl_saliency.select.base import IndexSelector


class AbsoluteIndex(IndexSelector):
    """
    Selects token index based on an absolute index (as opposed to relative to the generated tokens).

    Attributes:
        index (int): The absolute index of the token to select.
    """

    def __init__(self, index: int):
        self.index = index

    def select(self, view: SaliencyView) -> int:
        return self.index - view.gen_start_idx

    def __repr__(self) -> str:
        return f"AbsoluteIndex(index={self.index})"


class ReverseIndex(IndexSelector):
    """
    Selects a token based on a reverse index (counting from the end).

    Attributes:
        offset_from_end (int): The reverse index of the token to select (0-based from the end).
    """

    def __init__(self, offset_from_end: int):
        if offset_from_end < 0:
            raise IndexError(f"Reverse index must be non-negative, got {offset_from_end}.")
        self.offset_from_end = offset_from_end

    def select(self, view: SaliencyView) -> int:
        return view.num_tokens - 1 - self.offset_from_end

    def __repr__(self) -> str:
        return f"ReverseIndex(offset_from_end={self.offset_from_end})"
