"""Extract Localization heads. Modified from https://github.com/seilk/LocalizationHeads (CVPR 2025, Kang et al.)"""

import numpy as np
import torch
from scipy.ndimage import label


def _spatial_entropy(attn: torch.Tensor, threshold: float = 0.001) -> tuple[float, np.ndarray, int]:
    """Calculate spatial entropy of an attention map.

    Args:
        attn: 2D attention map tensor with shape [H, W].
        threshold: Binarization threshold for connected component analysis.

    Returns:
        result (tuple[float, np.ndarray, int]):
            - spatial_entropy (float): Entropy value (lower is better, inf if no mass).
            - labeled_mask (np.ndarray): Labeled connected components.
            - num_components (int): Number of connected components found.
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
        return float("inf"), labeled_mask, 0

    # Probability mass per component
    probs = [high_attn[labeled_mask == i].sum().item() / total_mass for i in range(1, num_components + 1)]

    # Calculate spatial entropy
    se = -sum(p * np.log(p) for p in probs) if probs else 0.0
    return se, labeled_mask, num_components


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
    line_len = np.linalg.norm(line)

    # Handle degenerate case
    if line_len == 0:
        return float(y[0])

    # Compute distances from points to line
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)

    # Elbow point is where distance is maximized
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def retrieve_localization_heads(
    attn: torch.Tensor,
    patch_size: tuple[int, int],
    *,
    chord_thresholding: bool = True,
    min_keep: int = 1,
    max_keep: int | None = 5,
) -> list[tuple[int, int]]:
    """Analyze heads and return a ranked list.

    Args:
        attn (torch.Tensor): Attention tensor with shape [L, H, P] where
            L is layers, H is heads, and P is patches.
        patch_size (tuple[int, int]): Tuple (H, W) indicating the height and width of the patches.
        chord_thresholding (bool, default=True): Whether to use chord method for thresholding.
        min_keep (int, default=1): Minimum number of heads to keep in the result.
        max_keep (int | None, default=5): Maximum number of heads to keep in the result. If None, keep all.
    Returns:
        (list[tuple[int, int]]):
            List of tuples (layer, head) containing analysis
            results for each head, sorted by spatial entropy (ascending).
    """
    layers, heads, _ = attn.shape

    # Criterion 1: Sum of attention values per head
    head_attns_sums = [attn[layer, head].sum().item() for layer in range(layers) for head in range(heads)]
    threshold = _elbow_chord(head_attns_sums) if chord_thresholding else min(head_attns_sums)

    # Analyze Criterion 2 only for heads above threshold (by value)
    candidates: list[dict] = []
    for idx, head_attn_sum in enumerate(head_attns_sums):
        # Compute layer and head indices
        layer = idx // heads
        head = idx % heads

        # Only analyze spatial entropy if above threshold
        if head_attn_sum >= threshold:
            # Compute spatial entropy
            attn_map = attn[layer, head].reshape(patch_size)  # [H, W]
            se, _, _ = _spatial_entropy(attn_map)

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
    kept = [head for head in candidates if np.isfinite(head["spatial_entropy"]) and head["layer"] > 1]

    # Fallback: Ensure minimum number of heads kept by taking top by attention sum
    if len(kept) < min_keep:
        rest = [head for head in candidates if not np.isfinite(head["spatial_entropy"]) or head["layer"] <= 1]
        rest.sort(key=lambda x: x["attn_sum"], reverse=True)
        kept.extend(rest[: min_keep - len(kept)])

    # Sort by spatial entropy (ascending)
    kept.sort(key=lambda x: x["spatial_entropy"])

    # Take only up to max_keep heads
    if max_keep and max_keep > 0:
        kept = kept[:max_keep]

    return [(res["layer"], res["head"]) for res in kept]
