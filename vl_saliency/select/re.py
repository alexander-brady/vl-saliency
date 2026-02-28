import re
from typing import Literal

from vl_saliency.core.scoped import ScopedSaliencyGrid


class ReSelector:
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
        select: Literal["first", "last"] = "first",
    ):
        if require_exact_match:
            pattern = pattern.lstrip("^").rstrip("$")  # Avoid double anchoring
            pattern = f"^{pattern}$"
        self.pattern = re.compile(pattern, flags)
        self.select = select

    def __call__(self, scoped: ScopedSaliencyGrid) -> int:
        """Selects a token index from the scoped saliency grid based on the regex pattern.

        Args:
            scoped (ScopedSaliencyGrid): The scoped saliency grid to select from.

        Returns:
            int: The index of the selected token.

        Raises:
            ValueError: If no tokens match the given pattern.
        """

        if scoped.input_ids is None or scoped._tok is None:
            raise ValueError("Input IDs and tokenizer are required for ReSelector to function.")

        tokens = scoped.decoded_gen_tokens

        # Determine search order based on selection preference
        search_tokens = tokens if self.select == "first" else reversed(tokens)

        try:
            idx = next(idx for idx, token in enumerate(search_tokens) if self.pattern.search(token))
        except StopIteration as e:
            raise ValueError(f"No tokens match the pattern: {self.pattern.pattern}") from e

        # Map back to original generated index space
        return idx if self.select == "first" else len(tokens) - 1 - idx
