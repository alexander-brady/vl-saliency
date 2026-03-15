"""Extract Localization heads. Modified from https://github.com/seilk/LocalizationHeads (CVPR 2025, Kang et al.)"""

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

try:
    from scipy.ndimage import label
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for connected component analysis. Please install scipy to use LocalizationHeads."
    ) from e


def _spatial_entropy(attn: torch.Tensor, threshold: float = 0.001) -> float:
    """Calculate spatial entropy of an attention map.

    Args:
        attn: 2D attention map tensor with shape [H, W].
        threshold: Binarization threshold for connected component analysis.

    Returns:
        float: The spatial entropy value (lower is better, inf if no mass).
    """

    # Emphasize regions significantly above the mean
    mean = torch.mean(attn)
    high_attn = torch.relu(attn - 2 * mean)

    # Compute connected components
    binary_mask = (high_attn > threshold).cpu().numpy().astype(np.int32)
    labeled_mask, num_components = label(binary_mask, structure=np.ones((3, 3)))  # type: ignore

    # Ensure there is some attention mass
    total_mass = high_attn.sum().item()
    if total_mass <= 0:
        return float("inf")

    # Probability mass per component
    probs = [
        high_attn[labeled_mask == i].sum().item() / total_mass for i in range(1, num_components + 1)
    ]

    # Calculate spatial entropy
    se = -sum(p * np.log(p) for p in probs) if probs else 0.0
    return se


def _elbow_chord(values: list[float]) -> float:
    """Find elbow point using perpendicular distance from chord method.

    Args:
        values: List of numeric values to analyze.

    Returns:
        float: The threshold value (y-coordinate) at the elbow point, not the index.
            Returns minimum value if 2 or fewer values provided.
    """
    if len(values) <= 2:
        return min(values) if values else 0.0

    # Ascending sort of values
    y: np.ndarray = np.array(sorted(values), dtype=np.float64)
    x: np.ndarray = np.arange(len(y), dtype=np.float64)

    # Line from first to last point
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)  # Always > 0 since len(values) > 2

    # Compute distances from points to line
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)

    # Elbow point is where distance is maximized
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def _identify_localization_heads(
    tensor: Float[Tensor, "layers heads H W"],
    chord_thresholding: bool = True,
    min_keep: int = 1,
    max_keep: int | None = 5,
) -> list[tuple[int, int]]:
    head_count = tensor.shape[1]

    # Criterion 1: Sum of attention values per head
    head_attn_sums = tensor.sum(dim=(-1, -2)).flatten().cpu().numpy().tolist()  # [layers, heads]
    threshold = _elbow_chord(head_attn_sums) if chord_thresholding else min(head_attn_sums)

    # Analyze Criterion 2 only for heads above threshold (by value)
    candidates: list[dict] = []
    for idx, head_attn_sum in enumerate(head_attn_sums):
        # Compute layer and head indices
        layer = idx // head_count
        head = idx % head_count

        # Only analyze spatial entropy if above threshold
        if head_attn_sum >= threshold:
            # Compute spatial entropy
            attn_map = tensor[layer, head]  # [H, W]
            se = _spatial_entropy(attn_map)

            # We want to avoid heads focusing on bottom row
            last_row_attended = (attn_map[-1] > 0.05).any()
            if last_row_attended:
                continue

        else:
            se = float("inf")

        candidates.append(
            {
                "layer": layer,
                "head": head,
                "attn_sum": head_attn_sum,
                "spatial_entropy": se,  # lower is better
            }
        )

    # Filter and sort: keep heads above threshold, prefer higher layers
    kept = [
        head for head in candidates if np.isfinite(head["spatial_entropy"]) and head["layer"] > 1
    ]

    # Fallback: Ensure minimum number of heads kept by taking top by attention sum
    if len(kept) < min_keep:
        rest = [
            head
            for head in candidates
            if not np.isfinite(head["spatial_entropy"]) or head["layer"] <= 1
        ]
        rest.sort(key=lambda x: x["attn_sum"], reverse=True)
        kept.extend(rest[: min_keep - len(kept)])

    # Sort by spatial entropy (ascending)
    kept.sort(key=lambda x: x["spatial_entropy"])

    # Take only up to max_keep heads
    if max_keep and max_keep > 0:
        kept = kept[:max_keep]

    if not kept:
        raise ValueError("No heads were selected by the LocalizationHeads criteria.")

    # Extract selected heads
    return [(head["layer"], head["head"]) for head in kept]


def extract_localization_heads(
    map: Float[Tensor, "layers heads H W"],
    chord_thresholding: bool = True,
    min_keep: int = 1,
    max_keep: int | None = 5,
) -> Float[Tensor, "kept_heads H W"]:
    """Extract localization heads from a saliency map.

    Args:
        map: Saliency map tensor with shape [layers, heads, H, W]
        chord_thresholding: Whether to use chord method for thresholding.
        min_keep: Minimum number of heads to keep in the result.
        max_keep: Maximum number of heads to keep in the result. If None, keep all.

    Returns:
        Tensor: Processed saliency map with shape [kept_heads, H, W].
    """
    heads_to_select = _identify_localization_heads(
        map,
        chord_thresholding=chord_thresholding,
        min_keep=min_keep,
        max_keep=max_keep,
    )

    layer_idx, head_idx = zip(*heads_to_select, strict=False)
    selected_heads = map[layer_idx, head_idx]  # shape: [kept_heads, H, W]
    return selected_heads
