"""
Histogram utilities for thumbnails (cached).
"""
from typing import Any, Dict, Optional
import numpy as np
from app.services.dsa_client import DSAClient
from app.services.image_fetching import get_dsa_client, _fetch_thumbnail_uncached
from app.services.ppc_cache import memory


@memory.cache
def _compute_color_histogram_cached(item_id: str, width: int, token: str, bins: int = 256) -> Dict[str, Any]:
    """
    Internal cached function to compute color histogram.
    Token is part of cache key to handle different authentication contexts.
    """
    img = _fetch_thumbnail_uncached(item_id, width, token)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")

    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalize to 0-255 range if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # Mask out background/white pixels for IHC analysis
    background_threshold = 240
    mask = ~((img[:, :, 0] > background_threshold) &
             (img[:, :, 1] > background_threshold) &
             (img[:, :, 2] > background_threshold))

    r_channel = img[:, :, 0][mask]
    g_channel = img[:, :, 1][mask]
    b_channel = img[:, :, 2][mask]

    hist_r = np.histogram(r_channel, bins=bins, range=(0, 256))[0] if len(r_channel) > 0 else np.zeros(bins)
    hist_g = np.histogram(g_channel, bins=bins, range=(0, 256))[0] if len(g_channel) > 0 else np.zeros(bins)
    hist_b = np.histogram(b_channel, bins=bins, range=(0, 256))[0] if len(b_channel) > 0 else np.zeros(bins)

    mean_r = float(np.mean(r_channel)) if len(r_channel) > 0 else 0.0
    mean_g = float(np.mean(g_channel)) if len(g_channel) > 0 else 0.0
    mean_b = float(np.mean(b_channel)) if len(b_channel) > 0 else 0.0

    std_r = float(np.std(r_channel)) if len(r_channel) > 0 else 0.0
    std_g = float(np.std(g_channel)) if len(g_channel) > 0 else 0.0
    std_b = float(np.std(b_channel)) if len(b_channel) > 0 else 0.0

    total_pixels = img.shape[0] * img.shape[1]
    tissue_pixels = np.sum(mask)
    background_pixels = total_pixels - tissue_pixels

    return {
        "item_id": item_id,
        "width": width,
        "bins": bins,
        "histogram_r": hist_r.tolist(),
        "histogram_g": hist_g.tolist(),
        "histogram_b": hist_b.tolist(),
        "statistics": {
            "mean": {"r": mean_r, "g": mean_g, "b": mean_b},
            "std": {"r": std_r, "g": std_g, "b": std_b},
        },
        "tissue_analysis": {
            "total_pixels": int(total_pixels),
            "tissue_pixels": int(tissue_pixels),
            "background_pixels": int(background_pixels),
            "tissue_percentage": float((tissue_pixels / total_pixels) * 100) if total_pixels > 0 else 0.0,
            "background_threshold": background_threshold,
        },
    }


def compute_color_histogram(
    item_id: str,
    width: int = 1024,
    bins: int = 256,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """Compute color histogram for an image thumbnail (cached)."""
    if dsa_client is None:
        dsa_client = get_dsa_client()
    token = dsa_client.get_token() or ""
    return _compute_color_histogram_cached(item_id, width, token, bins)

