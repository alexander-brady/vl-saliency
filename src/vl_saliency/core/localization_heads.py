"""Head analysis utilities. Modified from https://github.com/seilk/LocalizationHeads (CVPR 2025, Kang et al.)"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label, gaussian_filter


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
    mean = torch.mean(attn)
    
    # Emphasize regions significantly above the mean
    high_attn = F.relu(attn - mean * 2)
    
    # Compute connected components
    binary_mask = (high_attn > threshold).cpu().numpy().astype(np.int32)
    labeled_mask, num_components = label(binary_mask, structure=np.ones((3, 3)))  # type: ignore

    total_mass = high_attn.sum().item()
    if total_mass <= 0:
        return float("inf"), labeled_mask, 0
    
    # Probability mass per component
    probs = [
        high_attn[labeled_mask == i].sum().item() / total_mass
        for i in range(1, num_components + 1)
    ]
    
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
    y = np.array(sorted(values), dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    
    # Line from first to last point
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)

    # Handle degenerate case
    if line_len == 0:
        return y[0]

    # Compute distances from points to line
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)

    # Elbow point is where distance is maximized
    elbow_i = int(np.argmax(d))
    return y[elbow_i]


def retrieve_localization_heads(
    attn: torch.Tensor,
    patch_size: tuple[int, int],
    chord_thresholding: bool = True, 
    min_keep: int = 1, 
    max_keep: int = 5,
) -> list[tuple[int, int]]:
    """Analyze heads and return a ranked list.

    Args:
        attn (torch.Tensor): Attention tensor with shape [L, H, P] where 
            L is layers, H is heads, and P is patches.
        patch_size (tuple[int, int]): Tuple (H, W) indicating the height and width of the patches.
        chord_thresholding (bool, default=True): Whether to use chord method for thresholding.
        min_keep (int, default=1): Minimum number of heads to keep in the result.
        max_keep (int, default=5): Maximum number of heads to keep in the result.
    Returns:
        (list[tuple[int, int]]):
            List of tuples (layer, head) containing analysis
            results for each head, sorted by spatial entropy (ascending).
    """
    layers, heads, _ = attn.shape

    # Criterion 1: Sum of attention values per head
    head_attns = [
        attn[layer, head].sum().item()
        for layer in range(layers)
        for head in range(heads)
    ]
    threshold = _elbow_chord(head_attns) if chord_thresholding else min(head_attns)

    # Analyze Criterion 2 only for heads above threshold (by value)
    results: list[dict] = []
    heads_sum = iter(head_attns)
    for layer in range(layers):
        for head in range(heads):
            head_attn = next(heads_sum)
            if head_attn >= threshold:            
                attn_map = attn[layer, head].reshape(patch_size)  # [H, W]
                se, _, _ = _spatial_entropy(attn_map)

                # We want to avoid heads focusing on bottom row
                last_token_attended = (attn_map[-1] > 0.05).any()
                if not last_token_attended and se < float("inf"):
                    results.append({
                        "layer": layer,
                        "head": head,
                        "attn_sum": head_attn,
                        "spatial_entropy": se, # lower is better
                    })

    # Filter and sort: keep heads above threshold, prefer higher layers
    kept = [
        res for res in results
        if res["attn_sum"] >= threshold
        and res["layer"] > 1
    ][:max_keep]
    
    if len(kept) < min_keep: # fallback: take top by spatial entropy if too few
        kept = sorted(results, key=lambda x: x["attn_sum"], reverse=True)[:min_keep]
    kept.sort(key=lambda x: x["spatial_entropy"])  # ascending

    return [(res["layer"], res["head"]) for res in kept]


def _binarize_mean_relu(M: torch.Tensor) -> torch.Tensor:
    """Binarize a map using mean threshold and ReLU.

    Args:
        M: Input tensor to binarize.

    Returns:
        torch.Tensor: Binary tensor with dtype uint8.
    """
    m = M.mean()
    B = torch.relu(M - m)
    return (B > 0).to(torch.uint8)


# def upscale_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
#     # mask: [P, P] -> [H, W] using bilinear
#     P = mask.shape[0]
#     H, W = image_size[1], image_size[0]
#     t = torch.from_numpy(mask.astype(np.float32))[None, None]  # [1,1,P,P]
#     t_up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
#     return (t_up.detach().cpu().numpy() > 0.5).astype(np.uint8)


# def combine_heads(attn: torch.Tensor, selected: list[tuple[int, int]], P: int, sigma: float = 1.0) -> torch.Tensor:
    """Combine selected heads with optional Gaussian smoothing.

    Args:
        attn: Attention tensor with shape [L, H, 1, V] where L is layers,
            H is heads, and V is vocabulary/patches.
        selected: List of tuples (layer, head) specifying which attention
            heads to combine.
        P: Patch size determining the spatial dimensions of the output.
        sigma: Standard deviation for Gaussian smoothing. If sigma <= 0,
            no smoothing is applied.

    Returns:
        Combined 2D attention map as a torch.Tensor with shape [P, P]
        and dtype float32.
    """
    # M = torch.zeros((P, P), dtype=torch.float32, device=attn.device)
    # for l, h in selected:
    #     a2d = attn[l, h, 0].reshape(P, P).to(torch.float32)
    #     if sigma and sigma > 0:
    #         # Apply Gaussian smoothing via numpy
    #         a2d_np = a2d.detach().cpu().numpy()
    #         a2d_np = gaussian_filter(a2d_np, sigma=sigma)
    #         a2d = torch.from_numpy(a2d_np).to(device=attn.device, dtype=torch.float32)
    #     M += a2d
    # return _binarize_mean_relu(M)
