import html
from typing import Optional, Union, Sequence

import torch
from transformers import ProcessorMixin

try:
    from IPython.display import display, HTML  # optional
except Exception:  # pragma: no cover
    display = HTML = None


def render_token_ids(
    generated_ids: torch.Tensor,
    processor: ProcessorMixin,
    gen_start: int = 0,
    skip_tokens: Optional[Union[int, Sequence[int]]] = None,
    return_html: bool = False
) -> Optional[str]:
    """
    Visualizes the generated text from the model.
    
    Args:
        generated_ids (torch.Tensor): The generated token IDs.
        processor: The processor used to process input.
        gen_start (int): Index from which tokens are considered generated.
        skip_tokens (Optional[Union[int, List[int]]] = None): Token IDs to skip in the visualization.
        return_html (bool): If True, return the HTML string; otherwise display (if IPython available).

    Returns:
        HTML string if return_html=True, else None.
    """
    if generated_ids.dim() == 2:
        token_ids = generated_ids[0].tolist()
    else:
        token_ids = generated_ids.tolist()

    tok = processor.tokenizer
    tokens = tok.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
    
    skip_set = { skip_tokens } if isinstance(skip_tokens, int) else set(skip_tokens or [])
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    
    space_markers = ("▁", "Ġ")
    newline_markers = { "\n", "\\n", "Ċ", "▁\n" }

    # Styles
    FONTS = "font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;"
    COMMON = "display: inline-block; border-bottom: 1px solid #999; padding: 0 2px; margin: 1px; cursor: pointer;"
    PREFIX = "opacity: 0.4;"
    SPECIAL = "opacity: 0.6;"
    PROMPT = "background-color: #f5f5f5;"
    GENERATED = "background-color: #eafeee;"
    STYLE = (
        "<style>"
        ".token { transition: filter 0.2s ease; }"
        ".token:hover { filter: brightness(85%); }"
        "</style>"
    )

    buffer = [STYLE, f'<div style="{FONTS}">']
    for i, (token, tid) in enumerate(zip(tokens, token_ids)):
        if tid in skip_set:
            continue

        style = PROMPT if i < gen_start else GENERATED

        # Show faded leading space token (▁ or Ġ), if present
        if token.startswith(space_markers):
            fmt = f'<span style="{PREFIX}">{token[0]}</span>{html.escape(token[1:])}'
        elif tid in special_ids:
            fmt = f'<span style="{SPECIAL}">{html.escape(token)}</span>'
        else:
            fmt = html.escape(token)
            

        title = f"Token: {token} (ID: {tid})\nIndex: {i}"
        span = f'<div class="token" style="{COMMON} {style}" title="{html.escape(title)}">{fmt}</div>'
        buffer.append(span)
        
        if token in newline_markers:
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