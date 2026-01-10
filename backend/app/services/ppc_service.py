"""
PPC (Positive Pixel Count) service with caching
"""
import numpy as np
import logging
import requests
import os
import time
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional
from joblib import Memory
from app.services.dsa_client import DSAClient

# Try to import psutil for CPU metrics (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# Global DSA client instance (initialized on first use)
_dsa_client: Optional[DSAClient] = None


def get_dsa_client() -> DSAClient:
    """Get or create DSA client instance"""
    global _dsa_client
    if _dsa_client is None:
        _dsa_client = DSAClient()
    return _dsa_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize joblib Memory cache for numpy arrays
# Cache directory is persistent within Docker container
cache_dir = settings.CACHE_DIR
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


@memory.cache
def _fetch_thumbnail_uncached(item_id: str, width: int, token: str) -> Optional[np.ndarray]:
    """Internal cached function to fetch thumbnail (token is part of cache key)"""
    base_url = settings.DSA_BASE_URL.rstrip('/api/v1')
    url = f"{base_url}/api/v1/item/{item_id}/tiles/thumbnail?width={width}"
    if token:
        url += f"&token={token}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to get thumbnail for {item_id}: {e}")
        return None


def get_thumbnail_image(item_id: str, width: int = 1024, dsa_client: Optional[DSAClient] = None) -> Optional[np.ndarray]:
    """Fetch thumbnail image from DSA and return as numpy array (cached)"""
    if dsa_client is None:
        dsa_client = get_dsa_client()
    token = dsa_client.get_token() or ""
    return _fetch_thumbnail_uncached(item_id, width, token)


def _get_cpu_metrics() -> Dict[str, float]:
    """Get current CPU usage metrics"""
    if HAS_PSUTIL:
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_mb": 0.0}
    return {"cpu_percent": 0.0, "memory_mb": 0.0}


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
def _compute_ppc_hsi_cached(
    item_id: str,
    image_hash: str,
    hue_value: float = 0.1,  # Brown/DAB hue (normalized 0-1)
    hue_width: float = 0.1,  # Width of hue range
    saturation_minimum: float = 0.1,  # Minimum saturation
    intensity_upper_limit: float = 0.9,  # Above this = negative
    intensity_weak_threshold: float = 0.6,  # Separates weak from plain
    intensity_strong_threshold: float = 0.3,  # Separates plain from strong
    intensity_lower_limit: float = 0.05,  # Below this = negative
    thumbnail_width: int = 1024
) -> Dict[str, Any]:
    """
    Optimized HSI-based PPC computation (HistomicsTK-style).
    
    Uses HSI color space for more accurate IHC staining detection.
    Optimized to minimize memory copies and use vectorized operations.
    
    Args:
        item_id: DSA item ID
        image_hash: Hash for cache invalidation
        hue_value: Center hue for positive color (0-1, typically ~0.1 for brown/DAB)
        hue_width: Width of hue range (0-1)
        saturation_minimum: Minimum saturation for positive pixels (0-1)
        intensity_upper_limit: Intensity above which pixel is negative (0-1)
        intensity_weak_threshold: Intensity threshold separating weak from plain positive
        intensity_strong_threshold: Intensity threshold separating plain from strong positive
        intensity_lower_limit: Intensity below which pixel is negative (0-1)
        thumbnail_width: Width of thumbnail to process
    
    Returns:
        Dictionary with PPC results
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()
    
    try:
        dsa_client = get_dsa_client()
        img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
        
        if img is None:
            raise ValueError("Failed to retrieve thumbnail image")
        
        # Convert to RGB if needed (in-place view when possible)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # View, not copy
        
        # Normalize to 0-1 range (single conversion)
        if img.dtype == np.uint8:
            img = (img.astype(np.float32) / 255.0)
        else:
            img = img.astype(np.float32)
        
        img = np.clip(img, 0.0, 1.0)
        
        # Convert RGB to HSI (vectorized, optimized)
        hsi = rgb_to_hsi(img)
        h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]
        
        # Background mask (exclude very bright pixels)
        background_threshold = 0.94
        background_mask = (img[..., 0] > background_threshold) & \
                          (img[..., 1] > background_threshold) & \
                          (img[..., 2] > background_threshold)
        tissue_mask = ~background_mask
        
        # Apply tissue mask to HSI channels (use views, not copies)
        h_tissue = h[tissue_mask]
        s_tissue = s[tissue_mask]
        i_tissue = i[tissue_mask]
        
        if len(h_tissue) == 0:
            # No tissue found
            total_pixels = img.shape[0] * img.shape[1]
            return {
                "item_id": item_id,
                "total_pixels": total_pixels,
                "tissue_pixels": 0,
                "background_pixels": int(np.sum(background_mask)),
                "weak_positive_pixels": 0,
                "plain_positive_pixels": 0,
                "strong_positive_pixels": 0,
                "total_positive_pixels": 0,
                "weak_percentage": 0.0,
                "plain_percentage": 0.0,
                "strong_percentage": 0.0,
                "positive_percentage": 0.0,
                "method": "hsi",
                "parameters": {
                    "hue_value": hue_value,
                    "hue_width": hue_width,
                    "saturation_minimum": saturation_minimum,
                    "intensity_upper_limit": intensity_upper_limit,
                    "intensity_weak_threshold": intensity_weak_threshold,
                    "intensity_strong_threshold": intensity_strong_threshold,
                    "intensity_lower_limit": intensity_lower_limit,
                    "thumbnail_width": thumbnail_width,
                },
                "metrics": {
                    "execution_time_seconds": round(time.time() - start_time, 4),
                    "cpu_percent": 0.0,
                    "memory_mb": _get_cpu_metrics()["memory_mb"],
                }
            }
        
        # Positive pixel detection (vectorized, optimized)
        # Check hue range with wraparound (0.5 is center, so we check distance from hue_value)
        # HistomicsTK uses: abs(((hue - hue_value + 0.5) % 1) - 0.5) <= hue_width / 2
        hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
        hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
        
        # Check saturation and intensity bounds
        saturation_ok = s_tissue >= saturation_minimum
        intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
        
        # All positive pixels (meet all criteria)
        mask_all_positive = hue_in_range & saturation_ok & intensity_ok
        
        # Get intensities of positive pixels only (for classification)
        positive_intensities = i_tissue[mask_all_positive]
        
        if len(positive_intensities) == 0:
            # No positive pixels found
            total_tissue_pixels = int(np.sum(tissue_mask))
            total_pixels = img.shape[0] * img.shape[1]
            return {
                "item_id": item_id,
                "total_pixels": total_pixels,
                "tissue_pixels": total_tissue_pixels,
                "background_pixels": int(np.sum(background_mask)),
                "weak_positive_pixels": 0,
                "plain_positive_pixels": 0,
                "strong_positive_pixels": 0,
                "total_positive_pixels": 0,
                "weak_percentage": 0.0,
                "plain_percentage": 0.0,
                "strong_percentage": 0.0,
                "positive_percentage": 0.0,
                "method": "hsi",
                "parameters": {
                    "hue_value": hue_value,
                    "hue_width": hue_width,
                    "saturation_minimum": saturation_minimum,
                    "intensity_upper_limit": intensity_upper_limit,
                    "intensity_weak_threshold": intensity_weak_threshold,
                    "intensity_strong_threshold": intensity_strong_threshold,
                    "intensity_lower_limit": intensity_lower_limit,
                    "thumbnail_width": thumbnail_width,
                },
                "metrics": {
                    "execution_time_seconds": round(time.time() - start_time, 4),
                    "cpu_percent": 0.0,
                    "memory_mb": _get_cpu_metrics()["memory_mb"],
                }
            }
        
        # Classify positive pixels into weak, plain, strong based on intensity
        # Weak: intensity >= intensity_weak_threshold (lighter staining)
        # Strong: intensity < intensity_strong_threshold (darker staining)
        # Plain: intensity_strong_threshold <= intensity < intensity_weak_threshold (medium staining)
        mask_weak = positive_intensities >= intensity_weak_threshold
        mask_strong = positive_intensities < intensity_strong_threshold
        mask_plain = ~(mask_weak | mask_strong)
        
        # Note: Label image generation is handled by get_ppc_label_image() function
        # We don't generate it here to avoid storing large arrays in cache
        
        # Count pixels
        weak_count = int(np.sum(mask_weak))
        plain_count = int(np.sum(mask_plain))
        strong_count = int(np.sum(mask_strong))
        total_positive = weak_count + plain_count + strong_count
        
        # Calculate percentages
        total_tissue_pixels = int(np.sum(tissue_mask))
        total_pixels = img.shape[0] * img.shape[1]
        
        weak_percentage = (weak_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        plain_percentage = (plain_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        strong_percentage = (strong_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        positive_percentage = (total_positive / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        cpu_end = _get_cpu_metrics()
        cpu_delta = {
            "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
            "memory_mb": cpu_end["memory_mb"]
        }
        
        # Store label image in cache (as numpy array, will be converted to image on demand)
        # We'll return a flag indicating label image is available, and provide separate endpoint to fetch it
        
        return {
            "item_id": item_id,
            "total_pixels": total_pixels,
            "tissue_pixels": total_tissue_pixels,
            "background_pixels": int(np.sum(background_mask)),
            "weak_positive_pixels": weak_count,
            "plain_positive_pixels": plain_count,
            "strong_positive_pixels": strong_count,
            "total_positive_pixels": total_positive,
            "weak_percentage": round(weak_percentage, 2),
            "plain_percentage": round(plain_percentage, 2),
            "strong_percentage": round(strong_percentage, 2),
            "positive_percentage": round(positive_percentage, 2),
            "method": "hsi",
            "parameters": {
                "hue_value": hue_value,
                "hue_width": hue_width,
                "saturation_minimum": saturation_minimum,
                "intensity_upper_limit": intensity_upper_limit,
                "intensity_weak_threshold": intensity_weak_threshold,
                "intensity_strong_threshold": intensity_strong_threshold,
                "intensity_lower_limit": intensity_lower_limit,
                "thumbnail_width": thumbnail_width,
            },
            "metrics": {
                "execution_time_seconds": round(execution_time, 4),
                "cpu_percent": round(cpu_delta["cpu_percent"], 2),
                "memory_mb": round(cpu_delta["memory_mb"], 2),
            }
        }
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"HSI PPC computation failed for {item_id} after {execution_time:.4f}s: {e}", exc_info=True)
        raise


@memory.cache
def _compute_ppc_cached(
    item_id: str,
    image_hash: str,
    brown_threshold: float = 0.15,
    yellow_threshold: float = 0.20,
    red_threshold: float = 0.30,
    thumbnail_width: int = 1024
) -> Dict[str, Any]:
    """
    Internal cached PPC computation function.
    Image hash is used to invalidate cache when images change.
    This caches the expensive PPC computation.
    """
    # Start timing
    start_time = time.time()
    cpu_start = _get_cpu_metrics()
    
    try:
        # Get thumbnail image (this is already cached separately)
        dsa_client = get_dsa_client()
        img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
        
        if img is None:
            raise ValueError("Failed to retrieve thumbnail image")
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            # Grayscale, convert to RGB
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            # RGBA, convert to RGB
            img = img[:, :, :3]
        
        # Normalize to 0-1 range if needed
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Ensure image is in 0-1 range
        img = np.clip(img, 0.0, 1.0)
        
        # Extract RGB channels
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        
        # Create background mask (exclude white/very bright pixels)
        # Background pixels have all channels > threshold (default 240/255 = 0.94)
        background_threshold = 0.94
        background_mask = (r > background_threshold) & (g > background_threshold) & (b > background_threshold)
        tissue_mask = ~background_mask
        
        # Only analyze tissue pixels (exclude background)
        r_tissue = r[tissue_mask]
        g_tissue = g[tissue_mask]
        b_tissue = b[tissue_mask]
        
        # Compute RGB ratios for color classification
        # Avoid division by zero by adding small epsilon
        eps = 1e-6
        total_intensity = r_tissue + g_tissue + b_tissue + eps
        
        # RGB ratios (normalized)
        r_ratio = r_tissue / total_intensity
        g_ratio = g_tissue / total_intensity
        b_ratio = b_tissue / total_intensity
        
        # PPC Classification Logic (improved - more selective and mutually exclusive):
        # Brown/DAB: Low blue ratio (< 0.4), red and green roughly balanced
        # Yellow: High red+green (low blue), but red can be higher than green
        # Red: Very high red ratio, red dominates over green and blue
        
        # Brown pixels: low blue ratio AND red/green are roughly balanced (typical DAB)
        # brown_threshold: interpreted as minimum "brownness" = (1 - blue_ratio)
        # For brown/DAB, blue should be very low (< 0.3 typically)
        # Convert threshold: brown_threshold=0.15 means we want (1-blue) >= 0.15, so blue <= 0.85
        # But that's too permissive! Let's make it much more restrictive
        # Use brown_threshold directly as max blue ratio, but clamp to reasonable range
        # Default 0.15 -> use as max blue = 0.30 (much more selective)
        # Convert threshold: 0.15 -> max blue = 0.30 (more selective)
        max_blue_brown = min(0.35, max(0.20, brown_threshold * 2.0))
        # Brown: low blue, red and green balanced and significant
        brown_condition = (b_ratio < max_blue_brown) & \
                          (r_ratio >= 0.30) & \
                          (g_ratio >= 0.30) & \
                          (r_ratio < 0.50) & \
                          (g_ratio < 0.50) & \
                          (np.abs(r_ratio - g_ratio) < 0.12) & \
                          (r_ratio + g_ratio > 0.65)
        
        # Yellow pixels: high red+green (low blue), but red can be higher than green
        # yellow_threshold: minimum (red_ratio + green_ratio)
        # Default 0.20 means red+green >= 0.80, so blue <= 0.20
        max_blue_yellow = min(0.25, max(0.10, 1.0 - yellow_threshold * 3.0))
        # Yellow: low blue, red significant and higher than green
        yellow_condition = ((r_ratio + g_ratio) >= yellow_threshold) & \
                            (b_ratio < max_blue_yellow) & \
                            (r_ratio >= 0.40) & \
                            (r_ratio > g_ratio * 1.1) & \
                            ~brown_condition
        
        # Red pixels: very high red ratio, red dominates over green and blue
        red_condition = (r_ratio >= red_threshold) & \
                        (r_ratio > g_ratio * 1.3) & \
                        (r_ratio > b_ratio * 1.3) & \
                        ~brown_condition & \
                        ~yellow_condition
        
        # Count positive pixels (only in tissue regions)
        brown_pixels = int(np.sum(brown_condition))
        yellow_pixels = int(np.sum(yellow_condition))
        red_pixels = int(np.sum(red_condition))
        
        # Total tissue pixels (excluding background)
        total_tissue_pixels = int(np.sum(tissue_mask))
        total_pixels = img.shape[0] * img.shape[1]
        
        # Calculate percentages based on tissue pixels (not total pixels)
        brown_percentage = (brown_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        yellow_percentage = (yellow_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        red_percentage = (red_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        
        # Total positive pixels (union of brown, yellow, red - but typically they're mutually exclusive)
        # In practice, a pixel can be classified as multiple categories, so we count unique positive pixels
        positive_mask = brown_condition | yellow_condition | red_condition
        total_positive_pixels = int(np.sum(positive_mask))
        positive_percentage = (total_positive_pixels / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
        
        # End timing and compute metrics
        end_time = time.time()
        execution_time = end_time - start_time
        cpu_end = _get_cpu_metrics()
        
        # Compute CPU delta (approximate)
        cpu_delta = {
            "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
            "memory_mb": cpu_end["memory_mb"]
        }
        
        return {
            "item_id": item_id,
            "total_pixels": total_pixels,
            "tissue_pixels": total_tissue_pixels,
            "background_pixels": int(np.sum(background_mask)),
            "brown_pixels": brown_pixels,
            "yellow_pixels": yellow_pixels,
            "red_pixels": red_pixels,
            "total_positive_pixels": total_positive_pixels,
            "brown_percentage": round(brown_percentage, 2),
            "yellow_percentage": round(yellow_percentage, 2),
            "red_percentage": round(red_percentage, 2),
            "positive_percentage": round(positive_percentage, 2),
            "method": "rgb_ratio",
            "parameters": {
                "brown_threshold": brown_threshold,
                "yellow_threshold": yellow_threshold,
                "red_threshold": red_threshold,
                "thumbnail_width": thumbnail_width,
                "background_threshold": background_threshold,
            },
            "metrics": {
                "execution_time_seconds": round(execution_time, 4),
                "cpu_percent": round(cpu_delta["cpu_percent"], 2),
                "memory_mb": round(cpu_delta["memory_mb"], 2),
            }
        }
    except Exception as e:
        # Log timing even on error
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"PPC computation failed for {item_id} after {execution_time:.4f}s: {e}", exc_info=True)
        raise


def _get_image_hash(img: np.ndarray) -> str:
    """Generate a hash for an image array to use as cache key"""
    import hashlib
    return hashlib.md5(img.tobytes()).hexdigest()[:16]


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
        # Grayscale, convert to RGB
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        # RGBA, convert to RGB
        img = img[:, :, :3]
    
    # Normalize to 0-255 range if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Mask out background/white pixels for IHC analysis
    # Background is typically very bright (close to white)
    # Create a mask: exclude pixels that are too bright (likely background)
    # Threshold: exclude pixels where all channels are > threshold (e.g., 240)
    background_threshold = 240
    mask = ~((img[:, :, 0] > background_threshold) & 
             (img[:, :, 1] > background_threshold) & 
             (img[:, :, 2] > background_threshold))
    
    # Apply mask to each channel
    r_channel = img[:, :, 0][mask]
    g_channel = img[:, :, 1][mask]
    b_channel = img[:, :, 2][mask]
    
    # Compute histograms for each channel (only on tissue pixels)
    hist_r = np.histogram(r_channel, bins=bins, range=(0, 256))[0] if len(r_channel) > 0 else np.zeros(bins)
    hist_g = np.histogram(g_channel, bins=bins, range=(0, 256))[0] if len(g_channel) > 0 else np.zeros(bins)
    hist_b = np.histogram(b_channel, bins=bins, range=(0, 256))[0] if len(b_channel) > 0 else np.zeros(bins)
    
    # Compute statistics (only on tissue pixels)
    mean_r = float(np.mean(r_channel)) if len(r_channel) > 0 else 0.0
    mean_g = float(np.mean(g_channel)) if len(g_channel) > 0 else 0.0
    mean_b = float(np.mean(b_channel)) if len(b_channel) > 0 else 0.0
    
    std_r = float(np.std(r_channel)) if len(r_channel) > 0 else 0.0
    std_g = float(np.std(g_channel)) if len(g_channel) > 0 else 0.0
    std_b = float(np.std(b_channel)) if len(b_channel) > 0 else 0.0
    
    # Calculate percentage of tissue vs background
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
            "std": {"r": std_r, "g": std_g, "b": std_b}
        },
        "tissue_analysis": {
            "total_pixels": int(total_pixels),
            "tissue_pixels": int(tissue_pixels),
            "background_pixels": int(background_pixels),
            "tissue_percentage": float((tissue_pixels / total_pixels) * 100) if total_pixels > 0 else 0.0,
            "background_threshold": background_threshold
        }
    }


def compute_color_histogram(
    item_id: str,
    width: int = 1024,
    bins: int = 256,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Compute color histogram for an image thumbnail.
    
    Args:
        item_id: DSA item ID
        width: Width of thumbnail to process
        bins: Number of bins for histogram (default 256)
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with histogram data and statistics
    """
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    token = dsa_client.get_token() or ""
    return _compute_color_histogram_cached(item_id, width, token, bins)


def auto_threshold_ppc(
    item_id: str,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Automatically determine optimal thresholds for PPC computation.
    
    Uses histogram analysis to suggest brown, yellow, and red thresholds.
    
    Args:
        item_id: DSA item ID
        thumbnail_width: Width of thumbnail to process
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with suggested thresholds and metrics
    """
    # Start timing
    start_time = time.time()
    cpu_start = _get_cpu_metrics()
    
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    # Get thumbnail image
    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # Normalize to 0-1 range
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    img = np.clip(img, 0.0, 1.0)
    
    # Extract RGB channels
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    
    # Create background mask
    background_threshold = 0.94
    background_mask = (r > background_threshold) & (g > background_threshold) & (b > background_threshold)
    tissue_mask = ~background_mask
    
    # Only analyze tissue pixels
    r_tissue = r[tissue_mask]
    g_tissue = g[tissue_mask]
    b_tissue = b[tissue_mask]
    
    if len(r_tissue) == 0:
        # No tissue pixels found, return default thresholds
        return {
            "brown_threshold": 0.15,
            "yellow_threshold": 0.20,
            "red_threshold": 0.30
        }
    
    # Compute RGB ratios
    eps = 1e-6
    total_intensity = r_tissue + g_tissue + b_tissue + eps
    r_ratio = r_tissue / total_intensity
    g_ratio = g_tissue / total_intensity
    b_ratio = b_tissue / total_intensity
    
    # Auto-thresholding strategy:
    # 1. Brown: Use percentile of (1 - blue_ratio) distribution
    #    Higher percentile = more selective (fewer brown pixels)
    brown_metric = 1.0 - b_ratio
    brown_threshold = float(np.percentile(brown_metric, 75))  # 75th percentile
    
    # 2. Yellow: Use percentile of (red_ratio + green_ratio) distribution
    yellow_metric = r_ratio + g_ratio
    yellow_threshold = float(np.percentile(yellow_metric, 70))  # 70th percentile
    
    # 3. Red: Use percentile of red_ratio distribution
    red_threshold = float(np.percentile(r_ratio, 80))  # 80th percentile
    
    # Clamp thresholds to reasonable ranges
    brown_threshold = np.clip(brown_threshold, 0.10, 0.50)
    yellow_threshold = np.clip(yellow_threshold, 0.15, 0.60)
    red_threshold = np.clip(red_threshold, 0.20, 0.70)
    
    # End timing and compute metrics
    end_time = time.time()
    execution_time = end_time - start_time
    cpu_end = _get_cpu_metrics()
    
    cpu_delta = {
        "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
        "memory_mb": cpu_end["memory_mb"]
    }
    
    return {
        "brown_threshold": round(brown_threshold, 3),
        "yellow_threshold": round(yellow_threshold, 3),
        "red_threshold": round(red_threshold, 3),
        "metrics": {
            "execution_time_seconds": round(execution_time, 4),
            "cpu_percent": round(cpu_delta["cpu_percent"], 2),
            "memory_mb": round(cpu_delta["memory_mb"], 2),
        }
    }


def auto_detect_hue_parameters(
    item_id: str,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Automatically detect optimal hue_value and hue_width for HSI-based PPC.
    
    Analyzes the image to find the dominant brown/DAB hue and suggests parameters.
    
    Args:
        item_id: DSA item ID
        thumbnail_width: Width of thumbnail to process
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with suggested hue_value, hue_width, and other HSI parameters
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()
    
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # Normalize to 0-1 range
    if img.dtype == np.uint8:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    
    # Convert to HSI
    hsi = rgb_to_hsi(img)
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    
    # Background mask
    background_threshold = 0.94
    background_mask = (img[..., 0] > background_threshold) & \
                      (img[..., 1] > background_threshold) & \
                      (img[..., 2] > background_threshold)
    tissue_mask = ~background_mask
    
    # Get tissue pixels only
    h_tissue = h[tissue_mask]
    s_tissue = s[tissue_mask]
    i_tissue = i[tissue_mask]
    
    if len(h_tissue) == 0:
        # No tissue found, return defaults
        return {
            "hue_value": 0.1,
            "hue_width": 0.1,
            "saturation_minimum": 0.1,
            "intensity_upper_limit": 0.9,
            "intensity_weak_threshold": 0.6,
            "intensity_strong_threshold": 0.3,
            "intensity_lower_limit": 0.05,
            "metrics": {
                "execution_time_seconds": round(time.time() - start_time, 4),
                "cpu_percent": 0.0,
                "memory_mb": _get_cpu_metrics()["memory_mb"],
            }
        }
    
    # Filter for pixels with reasonable saturation and intensity (likely stained tissue)
    # Exclude very bright (background) and very dark (artifacts)
    valid_mask = (s_tissue > 0.1) & (i_tissue > 0.1) & (i_tissue < 0.9)
    h_valid = h_tissue[valid_mask]
    
    if len(h_valid) == 0:
        # No valid pixels, return defaults
        return {
            "hue_value": 0.1,
            "hue_width": 0.1,
            "saturation_minimum": 0.1,
            "intensity_upper_limit": 0.9,
            "intensity_weak_threshold": 0.6,
            "intensity_strong_threshold": 0.3,
            "intensity_lower_limit": 0.05,
            "metrics": {
                "execution_time_seconds": round(time.time() - start_time, 4),
                "cpu_percent": 0.0,
                "memory_mb": _get_cpu_metrics()["memory_mb"],
            }
        }
    
    # Find dominant hue (for brown/DAB, typically around 0.05-0.15 in normalized HSI)
    # Use histogram to find peak
    hist, bins = np.histogram(h_valid, bins=180)  # 180 bins = 2 degrees per bin
    peak_idx = np.argmax(hist)
    hue_value = float((bins[peak_idx] + bins[peak_idx + 1]) / 2.0)
    
    # Calculate hue width based on distribution
    # Use interquartile range (IQR) to determine width
    q25, q75 = np.percentile(h_valid, [25, 75])
    hue_width = float((q75 - q25) * 2.0)  # 2x IQR for reasonable coverage
    hue_width = np.clip(hue_width, 0.05, 0.3)  # Clamp to reasonable range
    
    # Suggest saturation minimum based on tissue saturation distribution
    saturation_minimum = float(np.percentile(s_tissue[valid_mask], 10))  # 10th percentile
    saturation_minimum = np.clip(saturation_minimum, 0.05, 0.3)
    
    # Intensity thresholds based on distribution
    intensity_upper_limit = float(np.percentile(i_tissue, 95))  # 95th percentile
    intensity_lower_limit = float(np.percentile(i_tissue, 5))   # 5th percentile
    intensity_weak_threshold = float(np.percentile(i_tissue[valid_mask], 60))  # 60th percentile
    intensity_strong_threshold = float(np.percentile(i_tissue[valid_mask], 30))  # 30th percentile
    
    # Clamp to reasonable ranges
    intensity_upper_limit = np.clip(intensity_upper_limit, 0.7, 0.98)
    intensity_lower_limit = np.clip(intensity_lower_limit, 0.02, 0.15)
    intensity_weak_threshold = np.clip(intensity_weak_threshold, 0.4, 0.8)
    intensity_strong_threshold = np.clip(intensity_strong_threshold, 0.15, 0.5)
    
    end_time = time.time()
    execution_time = end_time - start_time
    cpu_end = _get_cpu_metrics()
    cpu_delta = {
        "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
        "memory_mb": cpu_end["memory_mb"]
    }
    
    return {
        "hue_value": round(hue_value, 4),
        "hue_width": round(hue_width, 4),
        "saturation_minimum": round(saturation_minimum, 4),
        "intensity_upper_limit": round(intensity_upper_limit, 4),
        "intensity_weak_threshold": round(intensity_weak_threshold, 4),
        "intensity_strong_threshold": round(intensity_strong_threshold, 4),
        "intensity_lower_limit": round(intensity_lower_limit, 4),
        "metrics": {
            "execution_time_seconds": round(execution_time, 4),
            "cpu_percent": round(cpu_delta["cpu_percent"], 2),
            "memory_mb": round(cpu_delta["memory_mb"], 2),
        }
    }


def compute_ppc(
    item_id: str,
    brown_threshold: float = 0.15,
    yellow_threshold: float = 0.20,
    red_threshold: float = 0.30,
    thumbnail_width: int = 1024,
    method: str = "rgb_ratio",  # "rgb_ratio" or "hsi"
    # HSI parameters (only used if method="hsi")
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_weak_threshold: float = 0.6,
    intensity_strong_threshold: float = 0.3,
    intensity_lower_limit: float = 0.05,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Compute Positive Pixel Count (PPC) for an image.
    
    Supports two methods:
    - "rgb_ratio": Simple RGB ratio-based classification (faster, less accurate)
    - "hsi": HSI color space-based classification (HistomicsTK-style, more accurate)
    
    Args:
        item_id: DSA item ID
        brown_threshold: Threshold for brown pixels (0-1, RGB method only)
        yellow_threshold: Threshold for yellow pixels (0-1, RGB method only)
        red_threshold: Threshold for red pixels (0-1, RGB method only)
        thumbnail_width: Width of thumbnail to process
        method: "rgb_ratio" or "hsi" (default: "rgb_ratio")
        hue_value: Center hue for positive color (0-1, HSI method only)
        hue_width: Width of hue range (0-1, HSI method only)
        saturation_minimum: Minimum saturation (0-1, HSI method only)
        intensity_upper_limit: Intensity above which pixel is negative (0-1, HSI method only)
        intensity_weak_threshold: Intensity threshold for weak vs plain (0-1, HSI method only)
        intensity_strong_threshold: Intensity threshold for plain vs strong (0-1, HSI method only)
        intensity_lower_limit: Intensity below which pixel is negative (0-1, HSI method only)
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with PPC results
    """
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    # Get thumbnail to compute hash for cache invalidation
    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    
    # Compute image hash for cache invalidation
    img_hash = _get_image_hash(img)
    
    # Call appropriate cached PPC function
    if method == "hsi":
        return _compute_ppc_hsi_cached(
            item_id,
            img_hash,
            hue_value,
            hue_width,
            saturation_minimum,
            intensity_upper_limit,
            intensity_weak_threshold,
            intensity_strong_threshold,
            intensity_lower_limit,
            thumbnail_width
        )
    else:  # rgb_ratio (default)
        return _compute_ppc_cached(
            item_id,
            img_hash,
            brown_threshold,
            yellow_threshold,
            red_threshold,
            thumbnail_width
        )


def get_ppc_label_image(
    item_id: str,
    method: str = "hsi",
    thumbnail_width: int = 1024,
    # HSI parameters
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_weak_threshold: float = 0.6,
    intensity_strong_threshold: float = 0.3,
    intensity_lower_limit: float = 0.05,
    dsa_client: Optional[DSAClient] = None
) -> Optional[np.ndarray]:
    """
    Generate label image for HSI-based PPC visualization.
    
    Returns a label image where:
    - 0 = negative (background or non-positive tissue)
    - 1 = weak positive
    - 2 = plain positive  
    - 3 = strong positive
    
    Args:
        item_id: DSA item ID
        method: "hsi" (only HSI method supports label images currently)
        thumbnail_width: Width of thumbnail
        ... (HSI parameters)
        dsa_client: Optional DSA client instance
    
    Returns:
        Label image as numpy array (uint8), or None if method doesn't support it
    """
    if method != "hsi":
        return None  # Only HSI method supports label images currently
    
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        return None
    
    # Convert to RGB if needed
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # Normalize to 0-1 range
    if img.dtype == np.uint8:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    
    # Convert RGB to HSI
    hsi = rgb_to_hsi(img)
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]
    
    # Background mask
    background_threshold = 0.94
    background_mask = (img[..., 0] > background_threshold) & \
                      (img[..., 1] > background_threshold) & \
                      (img[..., 2] > background_threshold)
    tissue_mask = ~background_mask
    
    # Create label image
    label_image = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Positive pixel detection
    h_tissue = h[tissue_mask]
    s_tissue = s[tissue_mask]
    i_tissue = i[tissue_mask]
    
    if len(h_tissue) == 0:
        return label_image
    
    # Check hue range with wraparound
    hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
    hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
    
    # Check saturation and intensity bounds
    saturation_ok = s_tissue >= saturation_minimum
    intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
    
    # All positive pixels
    mask_all_positive = hue_in_range & saturation_ok & intensity_ok
    
    # Create full-size mask
    mask_all_positive_full = np.zeros(img.shape[:2], dtype=bool)
    mask_all_positive_full[tissue_mask] = mask_all_positive
    
    if np.any(mask_all_positive_full):
        # Get intensities for all positive pixels
        positive_intensities_full = i[mask_all_positive_full]
        
        # Classify based on intensity thresholds
        mask_weak_full = positive_intensities_full >= intensity_weak_threshold
        mask_strong_full = positive_intensities_full < intensity_strong_threshold
        mask_plain_full = ~(mask_weak_full | mask_strong_full)
        
        # Set labels
        positive_indices = np.where(mask_all_positive_full)
        weak_indices = (positive_indices[0][mask_weak_full], positive_indices[1][mask_weak_full])
        plain_indices = (positive_indices[0][mask_plain_full], positive_indices[1][mask_plain_full])
        strong_indices = (positive_indices[0][mask_strong_full], positive_indices[1][mask_strong_full])
        
        label_image[weak_indices] = 1  # WEAK
        label_image[plain_indices] = 2  # PLAIN
        label_image[strong_indices] = 3  # STRONG
    
    return label_image


@memory.cache
def _get_positive_pixel_intensities_cached(
    item_id: str,
    image_hash: str,
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_lower_limit: float = 0.05,
    thumbnail_width: int = 1024
) -> Dict[str, Any]:
    """
    Get positive pixel intensities without threshold classification.
    
    This allows frontend to reclassify in real-time when thresholds change.
    Cached separately from threshold classification (doesn't include weak/strong thresholds).
    
    Returns:
        Dictionary with:
        - positive_intensities: List of intensity values (0-1) for positive pixels
        - positive_pixel_indices: List of [row, col] pairs for positive pixels
        - tissue_pixels: Total tissue pixels
        - total_pixels: Total image pixels
        - background_pixels: Background pixel count
        - image_shape: [height, width]
    """
    start_time = time.time()
    
    try:
        dsa_client = get_dsa_client()
        img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
        
        if img is None:
            raise ValueError("Failed to retrieve thumbnail image")
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        # Normalize to 0-1 range
        if img.dtype == np.uint8:
            img = (img.astype(np.float32) / 255.0)
        else:
            img = img.astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        
        # Convert RGB to HSI
        hsi = rgb_to_hsi(img)
        h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]
        
        # Background mask
        background_threshold = 0.94
        background_mask = (img[..., 0] > background_threshold) & \
                          (img[..., 1] > background_threshold) & \
                          (img[..., 2] > background_threshold)
        tissue_mask = ~background_mask
        
        # Apply tissue mask to HSI channels
        h_tissue = h[tissue_mask]
        s_tissue = s[tissue_mask]
        i_tissue = i[tissue_mask]
        
        if len(h_tissue) == 0:
            return {
                "item_id": item_id,
                "positive_intensities": [],
                "positive_pixel_indices": [],
                "tissue_pixels": 0,
                "total_pixels": img.shape[0] * img.shape[1],
                "background_pixels": int(np.sum(background_mask)),
                "image_shape": [int(img.shape[0]), int(img.shape[1])],
                "metrics": {
                    "execution_time_seconds": round(time.time() - start_time, 4),
                }
            }
        
        # Positive pixel detection (without threshold classification)
        hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
        hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
        saturation_ok = s_tissue >= saturation_minimum
        intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
        mask_all_positive = hue_in_range & saturation_ok & intensity_ok
        
        # Get intensities and indices of positive pixels
        positive_intensities = i_tissue[mask_all_positive].tolist()
        
        # Get pixel indices (convert from tissue mask indices to full image indices)
        tissue_indices = np.where(tissue_mask)
        positive_tissue_indices = np.where(mask_all_positive)[0]
        positive_pixel_indices = [
            [int(tissue_indices[0][idx]), int(tissue_indices[1][idx])]
            for idx in positive_tissue_indices
        ]
        
        return {
            "item_id": item_id,
            "positive_intensities": positive_intensities,
            "positive_pixel_indices": positive_pixel_indices,
            "tissue_pixels": int(np.sum(tissue_mask)),
            "total_pixels": img.shape[0] * img.shape[1],
            "background_pixels": int(np.sum(background_mask)),
            "image_shape": [int(img.shape[0]), int(img.shape[1])],
            "metrics": {
                "execution_time_seconds": round(time.time() - start_time, 4),
            }
        }
    except Exception as e:
        logger.error(f"Failed to get positive pixel intensities for {item_id}: {e}", exc_info=True)
        raise


def get_positive_pixel_intensities(
    item_id: str,
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_lower_limit: float = 0.05,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Get positive pixel intensities without threshold classification.
    
    This allows frontend to reclassify in real-time when thresholds change.
    Cached separately from threshold classification.
    """
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    # Get thumbnail to compute hash for cache invalidation
    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    
    # Compute image hash for cache invalidation
    img_hash = _get_image_hash(img)
    
    return _get_positive_pixel_intensities_cached(
        item_id,
        img_hash,
        hue_value,
        hue_width,
        saturation_minimum,
        intensity_upper_limit,
        intensity_lower_limit,
        thumbnail_width
    )
