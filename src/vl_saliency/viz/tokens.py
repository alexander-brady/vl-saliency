import html
from collections.abc import Sequence
from typing import Literal, overload

from ..core._types import Processor

try:
    from IPython.display import HTML, display  # optional
except Exception:  # pragma: no cover
    display = HTML = None

_SPACE_MARKERS = ("▁", "Ġ")
_NEWLINE_MARKERS = {"\n", "\\n", "Ċ", "▁\n"}


@overload
def render_token_ids(
    token_ids: list[int],
    processor: Processor,
    return_html: Literal[True],
    gen_start: int = ...,
    skip_tokens: int | Sequence[int] | None = ...,
    only_number_generated: bool = ...,
) -> str: ...
@overload
def render_token_ids(
    token_ids: list[int],
    processor: Processor,
    return_html: Literal[False] = ...,
    gen_start: int = ...,
    skip_tokens: int | Sequence[int] | None = ...,
    only_number_generated: bool = ...,
) -> None: ...


def render_token_ids(
    token_ids: list[int],
    processor: Processor,
    return_html: bool = False,
    gen_start: int = 0,
    skip_tokens: int | Sequence[int] | None = None,
    only_number_generated: bool = False,
) -> str | None:
    """
    Visualizes the generated text from the model.

    Args:
        token_ids (list[int]): The generated token IDs.
        processor (Processor): The processor used to process input.
        gen_start (int): Index from which tokens are considered generated.
        skip_tokens (Optional[Union[int, List[int]]] = None): Token IDs to skip in the visualization.
        return_html (bool): If True, return the HTML string; otherwise display (if IPython available).
        only_number_generated (bool): If True, only number the generated tokens in the tooltip.

    Returns:
        HTML string if return_html=True, else None.
    """

    tok = processor.tokenizer
    tokens = tok.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    skip_set = {skip_tokens} if isinstance(skip_tokens, int) else set(skip_tokens or [])
    special_ids = set(getattr(tok, "all_special_ids", []) or [])

    # Styles
    FONTS = "font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;"
    COMMON = "display: inline-block; border-bottom: 1px solid #999; padding: 0 2px; margin: 1px; cursor: pointer;"
    PREFIX = "opacity: 0.4;"
    SPECIAL = "opacity: 0.6;"
    PROMPT = "background-color: #f5f5f5;"
    GENERATED = "background-color: #eafeee;"
    STYLE = "<style>.token { transition: filter 0.2s ease; }.token:hover { filter: brightness(85%); }</style>"

    buffer = [STYLE, f'<div style="{FONTS}">']
    for i, (token, tid) in enumerate(zip(tokens, token_ids, strict=False)):
        if tid in skip_set:
            continue

        style = PROMPT if i < gen_start else GENERATED

        # Show faded leading space token (▁ or Ġ), if present
        if token.startswith(_SPACE_MARKERS):
            fmt = f'<span style="{PREFIX}">{token[0]}</span>{html.escape(token[1:])}'
        elif tid in special_ids:
            fmt = f'<span style="{SPECIAL}">{html.escape(token)}</span>'
        else:
            fmt = html.escape(token)

        # Set tooltip content
        if only_number_generated:
            if i < gen_start:
                title = f"Token: {token} (ID: {tid})"
            else:
                title = f"Token: {token} (ID: {tid})\nIndex: {i - gen_start}"
        else:
            title = f"Token: {token} (ID: {tid})\nIndex: {i}"

        # Render token span
        span = f'<div class="token" style="{COMMON} {style}" title="{html.escape(title)}">{fmt}</div>'
        buffer.append(span)

        # Show line breaks for newline tokens
        if token in _NEWLINE_MARKERS:
            buffer.append("<br>")

    buffer.append("</div>")
    out = "".join(buffer)

    if return_html:
        return out
    if display and HTML:
        display(HTML(out))
    else:
        print(out)
    return None
