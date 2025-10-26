from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace


class Selector(ABC):
    """Base class for all selectors."""

    @abstractmethod
    def select(self, trace: Trace) -> int:
        pass
