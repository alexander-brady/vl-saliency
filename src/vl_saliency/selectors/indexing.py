from ..core.trace import Trace
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AbsoluteIndex:
    """Selector that turns a absolute token into relative index.

    Attributes:
        index (int): The absolute token index in the sequence.

    Raises:
        ValueError: If the absolute index refers to a non-generated token.
    """

    def __init__(self, index: int):
        self.index = index

    def __call__(self, trace: Trace) -> int:
        if trace.gen_start is None:
            logger.warning("Trace has no gen_start; using absolute index as-is.")
            return self.index

        if self.index < trace.gen_start:
            raise ValueError("AbsoluteIndex refers to a non-generated token.")

        return self.index - trace.gen_start


class IndexFromEnd:
    """
    Selector that selects a token relative to the end of generated tokens.

    Attributes:
        offset (int): The offset from the end of generated tokens, must be positive.

    Raises:
        ValueError: If offset is not positive or exceeds total generated tokens.
    """

    def __init__(self, offset: int):
        self.offset = offset

    def __call__(self, trace: Trace) -> int:
        if self.offset <= 0:
            raise ValueError("Offset must be positive.")

        if self.offset > trace.total_generated_tokens:
            raise ValueError("Offset exceeds total generated tokens.")

        return trace.total_generated_tokens - self.offset
