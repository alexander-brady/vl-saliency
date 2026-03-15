import re
from typing import Literal

from vl_saliency.maps.view import SaliencyView
from vl_saliency.select.base import IndexSelector


class RegexSelector(IndexSelector):
    """
    Selects tokens from based on a regular expression pattern.

    Attributes:
        pattern (str): The regex pattern to match tokens.
        flags (int): Regex flags from the re module.
        require_exact_match (bool): If True, the entire token must match the pattern.
        select (Literal['first', 'last']): Whether to select the first or last matching token.

    Raises:
        ValueError: If no tokens match the given pattern.
    """

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        require_exact_match: bool = False,
        occurrence: Literal["first", "last"] = "first",
    ):
        if require_exact_match:
            pattern = pattern.lstrip("^").rstrip("$")  # Avoid double anchoring
            pattern = f"^{pattern}$"
        self.pattern = re.compile(pattern, flags)
        self.occurrence = occurrence

    def select(self, view: SaliencyView) -> int:
        """Selects a token index from the view saliency grid based on the regex pattern.

        Args:
            view (viewSaliencyGrid): The view saliency grid to select from.

        Returns:
            int: The index of the selected token.

        Raises:
            ValueError: If no tokens match the given pattern.
        """

        tokens = view.decoded_gen_tokens

        if self.occurrence == "last":
            indices = range(len(tokens) - 1, -1, -1)
        else:
            indices = range(len(tokens))

        for i in indices:
            if self.pattern.search(tokens[i]):
                return i

        raise ValueError(f"No tokens match the pattern: {self.pattern.pattern}")

    def __repr__(self) -> str:
        return f"RegexSelector(pattern={self.pattern.pattern!r}, occurrence={self.occurrence})"
