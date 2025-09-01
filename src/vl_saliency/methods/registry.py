"""
Registry for saliency methods.

Lets you register a method under a string key and resolve it later:
```python
    from vl_saliency.methods.registry import register

    @register("gradcam", "gradattn")
    def make_gradattn(attn: torch.Tensor, grad: torch.Tensor):
        return ...

    method = resolve("gradcam")
```
"""

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}
_ALIASES: dict[str, str] = {}


def register(
    name: str, *aliases: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name.lower()] = func
        for alias in aliases:
            _ALIASES[alias.lower()] = name
        return func

    return decorator


def resolve(name: str) -> Callable[..., Any]:
    """Return a saliency method by name or alias."""
    name = name.lower()
    if name in _ALIASES:
        name = _ALIASES[name]
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise ValueError(f"Unknown saliency method: {name}")


def list_methods() -> list[str]:
    """Return list of registered method names (not aliases)."""
    return sorted(_REGISTRY.keys())


def list_aliases() -> dict[str, str]:
    """Return alias → canonical mapping."""
    return dict(_ALIASES)
