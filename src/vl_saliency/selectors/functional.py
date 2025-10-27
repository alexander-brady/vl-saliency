from ..core.trace import Trace
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AbsoluteIndex:
    """Selector that turns a absolute token into relative index."""

    def __init__(self, index: int):
        self.index = index

    def __call__(self, trace: Trace) -> int:
        if trace.gen_start is None:
            logger.warning("Trace has no gen_start; using absolute index as-is.")
            return self.index

        if self.index < trace.gen_start:
            raise ValueError("AbsoluteIndex refers to a non-generated token.")

        return self.index - trace.gen_start
