"""Head analysis utilities."""
from typing import Dict, List

import numpy as np
import torch
from typing import Dict, List, Tuple

from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

def spatial_entropy(attn_map_2d: torch.Tensor, threshold: float = 0.001) -> Dict:
    """Calculate spatial entropy of an attention map.

    Args:
        attn_map_2d: 2D attention map tensor with shape [P, P].
        threshold: Binarization threshold for connected component analysis.

    Returns:
        Dictionary containing:
            - spatial_entropy (float): Entropy value (lower is better, inf if no mass).
            - labeled_array (np.ndarray): Labeled connected components.
            - num_components (int): Number of connected components found.
    """
    S = attn_map_2d
    mean_val = torch.mean(S)
    B = torch.relu(S - mean_val*2)
    B_np = B.detach().cpu().to(torch.float32).numpy()
    binary = (B_np > threshold).astype(np.int32)

    from scipy.ndimage import label
    labeled, num = label(binary, structure=np.ones((3, 3)))

    total = float(B.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}

    # Probability mass per component
    probs = []
    for i in range(1, num + 1):
        comp_sum = B_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)

    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}


def elbow_chord(values: List[float]) -> float:
    """Find elbow point using perpendicular distance from chord method.

    Args:
        values: List of numeric values to analyze.

    Returns:
        The threshold value (y-coordinate) at the elbow point, not the index.
        Returns minimum value if 2 or fewer values provided.
    """
    if len(values) <= 2:
        return min(values) if values else 0.0
    
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)  # ascending
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    
    if line_len == 0:
        return y[0]
    
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def analyze_heads(attn: torch.Tensor, patch_size: int | None = None, chord_method: bool = True, min_keep: int = 1, max_keep: int = 5) -> List[tuple[int, int]]:
    """Analyze heads and return a ranked list.

    Args:
        attn: Attention tensor with shape [L, H, 1, V] where L is layers,
            H is heads, and V is vocabulary/patches.
        patch_size: Patch size (P) if known, else inferred as sqrt(V).
        chord_method: Whether to use chord method for thresholding.
        min_keep: Minimum number of heads to keep in the result.

    Returns:
        List of tuples (layer, head) containing analysis results for each head,
        sorted by spatial entropy.
    """
    L, H, _, V = attn.shape
    P = patch_size if patch_size else int(np.sqrt(V)) # TODO: Why do they use this value, while we use H*W in trace.py?

    # Criterion 1: head sums over image patches
    sums = []
    for l in range(L):
        for h in range(H):
            s = float(attn[l, h, 0].sum().item())
            sums.append(s)

    thr_val = elbow_chord(sums) if chord_method else min(sums)

    # Analyze Criterion 2 only for heads above thr_val (by value)
    results: List[Dict] = []
    idx = 0
    for l in range(L):
        for h in range(H):
            s = sums[idx]
            idx += 1
            if s < thr_val:
                se = float("inf")
                bottom_row_focus = False
                n_comp = 0
            else:
                a2d = attn[l, h, 0].reshape(P, P)
                se_res = spatial_entropy(a2d)
                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > 0.05).any())
                se = float(se_res["spatial_entropy"])    # lower is better
                labeled = se_res["labeled_array"]
                n_comp = int(se_res["num_components"])
            results.append({
                "layer": l,
                "head": h,
                "attn_sum": s,
                "spatial_entropy": se,
                "bottom_row_focus": bottom_row_focus,
                "num_components": n_comp,
            })

    # Filter and sort: keep heads above threshold, prefer non-bottom-row
    kept = [r for r in results if np.isfinite(r["spatial_entropy"]) and r["attn_sum"] >= thr_val and not r["bottom_row_focus"] and r["layer"] > 1]
    if len(kept) < min_keep:
        # fallback: take top by sum if too few
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: min_keep]
    if len(kept) > max_keep:
        kept = kept[: max_keep]

    kept.sort(key=lambda x: x["spatial_entropy"])  # ascending
    return [(r["layer"], r["head"]) for r in kept]

def binarize_mean_relu(M: torch.Tensor) -> torch.Tensor:
    """Binarize a map using mean threshold and ReLU.
    
    Args:
        M: Input tensor to binarize.
        
    Returns:
        Binary tensor with dtype uint8.
    """
    m = M.mean()
    B = torch.relu(M - m)
    return (B > 0).to(torch.uint8)

def upscale_mask(mask: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    # mask: [P, P] -> [H, W] using bilinear
    P = mask.shape[0]
    H, W = image_size[1], image_size[0]
    t = torch.from_numpy(mask.astype(np.float32))[None, None]  # [1,1,P,P]
    t_up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return (t_up.detach().cpu().numpy() > 0.5).astype(np.uint8)

def combine_heads(attn: torch.Tensor, selected: List[tuple[int, int]], P: int, sigma: float = 1.0) -> torch.Tensor:
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
    M = torch.zeros((P, P), dtype=torch.float32, device=attn.device)
    for l, h in selected:
        a2d = attn[l, h, 0].reshape(P, P).to(torch.float32)
        if sigma and sigma > 0:
            # Apply Gaussian smoothing via numpy
            a2d_np = a2d.detach().cpu().numpy()
            a2d_np = gaussian_filter(a2d_np, sigma=sigma)
            a2d = torch.from_numpy(a2d_np).to(device=attn.device, dtype=torch.float32)
        M += a2d
    return upscale_mask(binarize_mean_relu(M))