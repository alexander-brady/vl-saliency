from collections.abc import Iterator, Sequence
from dataclasses import dataclass

type SelectionSpec = LayerSelect | HeadSelect | Sequence[int] | Sequence[tuple[int, int]] | None
"""Specification of which layers or attention heads contribute to saliency computation.


Accepted forms:
- ``None``: use all layers and heads.
- ``Sequence[int]``: shorthand for selecting specific layers.
- ``Sequence[tuple[int, int]``: shorthand for selecting specific heads (layer, head) per layer.
- :class:`LayerSelect`: explicit layer selection.
- :class:`HeadSelect`: explicit head selection.
"""


@dataclass(frozen=True, init=False)
class LayerSelect:
    """
    Select a subset of transformer layers to include in saliency computation.

    Layers not present are excluded from saliency computation.

    Example:
        ``LayerSelect(0, 2)`` selects layers 0 and 2.
    """

    layers: tuple[int, ...]
    """Indices of layers to include."""

    def __init__(self, *layers: int):
        object.__setattr__(self, "layers", tuple(sorted(set(layers))))

    def __iter__(self) -> Iterator[int]:
        return iter(self.layers)


@dataclass(frozen=True, init=False)
class HeadSelect:
    """Select specific attention heads within particular layers.

    Heads not present are excluded from saliency computation.

    Example:
        ``HeadSelect((0, 0), (0, 2), (1, 1))`` selects heads 0 and 2 in layer 0,
        and head 1 in layer 1.
    """

    heads: dict[int, tuple[int]]
    """Mapping from layer index to the head indices selected in that layer."""

    def __init__(self, *heads: tuple[int, int]) -> None:
        grouped: dict[int, list[int]] = {}

        for layer, head in sorted(set(heads)):
            grouped.setdefault(layer, []).append(head)

        object.__setattr__(
            self,
            "heads",
            {layer: tuple(heads) for layer, heads in grouped.items()},
        )

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for layer, heads in self.items():
            for head in heads:
                yield layer, head

    def __contains__(self, item: tuple[int, int]) -> bool:
        layer, head = item
        return head in self.heads[layer]

    def items(self):
        return self.heads.items()
