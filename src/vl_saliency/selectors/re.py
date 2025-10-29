import re
from typing import Literal

from ..core.trace import Trace


class ReSelector:
    """
    Selects tokens from a trace based on a regular expression pattern.

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
            pattern = f"^{pattern}$"
        self.pattern = re.compile(pattern, flags)
        self.select = select

    def __call__(self, trace: Trace) -> int:
        if trace.processor is None:
            raise ValueError("Trace has no processor to provide text.")

        if trace.generated_ids is None:
            raise ValueError("Trace has no generated token IDs.")

        # Decode the generated IDs to text
        tok = trace.processor.tokenizer  # type: ignore[attr-defined]
        tokens = tok.convert_ids_to_tokens(trace.generated_ids, skip_special_tokens=False)

        # Only consider generated tokens
        generated_tokens = tokens[trace.gen_start :]

        # Determine search order based on 'select' attribute
        if self.select == "first":
            search_range = range(len(generated_tokens))
        else:  # self.select == "last"
            search_range = range(len(generated_tokens) - 1, -1, -1)
        print(tokens, generated_tokens)

        # Find the indices of matching tokens
        for i in search_range:
            token = generated_tokens[i]
            if self.pattern.search(token):
                return i

        raise ValueError("No tokens match the given pattern.")
