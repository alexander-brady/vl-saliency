from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.trace import Trace


@runtime_checkable
class Selector(Protocol):
    """Selector protocol that determines a token index from a Trace."""

    def __call__(self, trace: Trace) -> int: ...
