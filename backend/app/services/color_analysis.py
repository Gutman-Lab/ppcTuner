"""
Color space conversion and analysis utilities
"""
import numpy as np
from typing import Dict, Any, Optional
from joblib import Memory
import os
from app.core.config import settings

# Initialize joblib Memory cache
cache_dir = settings.CACHE_DIR
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


def rgb_to_hsi(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSI (Hue, Saturation, Intensity) color space.
    
    Optimized vectorized implementation - no loops, minimal memory copies.
    Based on HistomicsTK's color conversion.
    
    Args:
        rgb: RGB image array, shape (H, W, 3), values in [0, 1]
    
    Returns:
        HSI image array, shape (H, W, 3), values in [0, 1]
        Channels: [Hue, Saturation, Intensity]
    """
    # Ensure input is float32 for efficiency
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    
    # Intensity (I) = average of RGB
    intensity = (r + g + b) / 3.0
    
    # Avoid division by zero for black pixels
    eps = 1e-6
    total = r + g + b + eps
    
    # Normalized RGB
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    
    # Saturation calculation
    # S = 1 - 3 * min(R, G, B) / (R + G + B)
    min_rgb = np.minimum(np.minimum(r_norm, g_norm), b_norm)
    saturation = 1.0 - 3.0 * min_rgb
    
    # Hue calculation (handles wraparound at 0/360 degrees)
    # H = arccos((0.5 * ((R - G) + (R - B))) / sqrt((R - G)^2 + (R - B)(G - B)))
    numerator = 0.5 * ((r_norm - g_norm) + (r_norm - b_norm))
    denominator = np.sqrt((r_norm - g_norm) ** 2 + (r_norm - b_norm) * (g_norm - b_norm)) + eps
    
    # Clamp to [-1, 1] for arccos
    cos_h = np.clip(numerator / denominator, -1.0, 1.0)
    hue = np.arccos(cos_h)
    
    # Adjust hue based on blue component
    # If B > G, hue = 2π - hue (wraparound)
    hue = np.where(b_norm > g_norm, 2.0 * np.pi - hue, hue)
    
    # Normalize hue to [0, 1] range (divide by 2π)
    hue = hue / (2.0 * np.pi)
    
    # Ensure saturation is in [0, 1]
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # Stack into HSI array (no copy, just view manipulation)
    hsi = np.stack([hue, saturation, intensity], axis=-1)
    
    return hsi


@memory.cache
def _compute_color_histogram_cached(item_id: str, width: int, token: str, bins: int = 256) -> Dict[str, Any]:
    """Internal cached function to compute color histogram"""
    from app.services.image_fetching import get_thumbnail_image, get_dsa_client
    
    dsa_client = get_dsa_client()
    img = get_thumbnail_image(item_id, width=width, dsa_client=dsa_client)
    
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # Normalize to 0-255 range if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Compute histograms for each channel
    hist_r, _ = np.histogram(img[:, :, 0].flatten(), bins=bins, range=(0, 256))
    hist_g, _ = np.histogram(img[:, :, 1].flatten(), bins=bins, range=(0, 256))
    hist_b, _ = np.histogram(img[:, :, 2].flatten(), bins=bins, range=(0, 256))
    
    return {
        "histogram_r": hist_r.tolist(),
        "histogram_g": hist_g.tolist(),
        "histogram_b": hist_b.tolist(),
        "bins": bins,
    }


def compute_color_histogram(item_id: str, width: int = 1024, bins: int = 256, dsa_client: Optional[Any] = None) -> Dict[str, Any]:
    """
    Compute RGB color histogram for an image.
    
    Args:
        item_id: DSA item ID
        width: Width of thumbnail to process
        bins: Number of histogram bins (default: 256)
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with histogram data
    """
    from app.services.image_fetching import get_dsa_client
    
    if dsa_client is None:
        dsa_client = get_dsa_client()
    token = dsa_client.get_token() or ""
    return _compute_color_histogram_cached(item_id, width, token, bins)
