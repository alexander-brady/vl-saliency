from abc import ABC, abstractmethod

from ..core.trace import Trace


class Selector(ABC):
    """Base class for all selectors."""

    @abstractmethod
    def select(self, trace: Trace) -> int:
        pass
