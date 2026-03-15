from .pos import AbsoluteIndex, ReverseIndex
from .regex import RegexSelector


def regex(pattern: str, **kwargs) -> RegexSelector:
    """Find token that matches regex pattern."""
    return RegexSelector(pattern, **kwargs)


def absolute(index: int) -> AbsoluteIndex:
    """Number tokens from beginning (i.e. including image, padding, etc)."""
    return AbsoluteIndex(index)


def from_end(offset: int = 0) -> ReverseIndex:
    """Number tokens starting from the last one."""
    return ReverseIndex(offset)
