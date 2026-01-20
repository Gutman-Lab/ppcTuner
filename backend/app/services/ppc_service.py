"""
PPC (Positive Pixel Count) service with caching
"""
import numpy as np
import logging
import time
from typing import Dict, Any, Optional
from app.services.dsa_client import DSAClient
from app.services.image_fetching import get_thumbnail_image, get_region_image, get_dsa_client
from app.services.color_analysis import rgb_to_hsi
from app.services.ppc_cache import memory
from app.services.ppc_metrics import _get_cpu_metrics
from app.services.ppc_utils import _get_image_hash
from app.services.ppc_histogram import compute_color_histogram
from app.services.ppc_thresholds import auto_threshold_ppc, auto_detect_hue_parameters, auto_detect_hue_parameters_sampled_rois
from app.services.ppc_label_images import (
    get_ppc_label_image,
    get_ppc_label_image_region,
    get_positive_pixel_intensities,
)

logger = logging.getLogger(__name__)


def _analyze_ppc_hsi(
    img: np.ndarray,
    item_id: str,
    hue_value: float,
    hue_width: float,
    saturation_minimum: float,
    intensity_upper_limit: float,
    intensity_weak_threshold: float,
    intensity_strong_threshold: float,
    intensity_lower_limit: float,
    extra_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Common PPC analysis logic for HSI method.
    
    This function performs the actual PPC computation on an image array.
    It's used by both thumbnail and region analysis - only the image fetch differs.
    
    Args:
        img: Image array (numpy array, can be uint8 or float32)
        item_id: DSA item ID (for result metadata)
        hue_value: Center hue for positive color (0-1)
        hue_width: Width of hue range (0-1)
        saturation_minimum: Minimum saturation (0-1)
        intensity_upper_limit: Intensity above which pixel is negative (0-1)
        intensity_weak_threshold: Intensity threshold for weak vs plain (0-1)
        intensity_strong_threshold: Intensity threshold for plain vs strong (0-1)
        intensity_lower_limit: Intensity below which pixel is negative (0-1)
        extra_params: Optional dict to merge into parameters (e.g., region coordinates)
    
    Returns:
        Dictionary with PPC results
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()
    
    try:
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
            params = {
                "hue_value": hue_value,
                "hue_width": hue_width,
                "saturation_minimum": saturation_minimum,
                "intensity_upper_limit": intensity_upper_limit,
                "intensity_weak_threshold": intensity_weak_threshold,
                "intensity_strong_threshold": intensity_strong_threshold,
                "intensity_lower_limit": intensity_lower_limit,
            }
            if extra_params:
                params.update(extra_params)
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
                "parameters": params,
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
            params = {
                "hue_value": hue_value,
                "hue_width": hue_width,
                "saturation_minimum": saturation_minimum,
                "intensity_upper_limit": intensity_upper_limit,
                "intensity_weak_threshold": intensity_weak_threshold,
                "intensity_strong_threshold": intensity_strong_threshold,
                "intensity_lower_limit": intensity_lower_limit,
            }
            if extra_params:
                params.update(extra_params)
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
                "parameters": params,
                "metrics": {
                    "execution_time_seconds": round(time.time() - start_time, 4),
                    "cpu_percent": 0.0,
                    "memory_mb": _get_cpu_metrics()["memory_mb"],
                }
            }
        
        # Classify positive pixels into weak, plain, strong based on intensity
        # 
        # IMPORTANT: In HSI color space, "intensity" refers to pixel BRIGHTNESS (0=dark, 1=bright).
        # This is the opposite of "staining intensity"!
        #
        # For IHC staining (DAB/brown):
        #   - Strong staining = lots of DAB = dark brown = LOW pixel intensity (closer to 0)
        #   - Weak staining = little DAB = light brown = HIGH pixel intensity (closer to 1)
        #   - Negative = white background = VERY HIGH pixel intensity (close to 1)
        #
        # Classification:
        #   Weak: intensity >= intensity_weak_threshold (light brown, high pixel brightness)
        #   Strong: intensity < intensity_strong_threshold (dark brown, low pixel brightness)
        #   Plain: intensity_strong_threshold <= intensity < intensity_weak_threshold (medium brown)
        mask_weak = positive_intensities >= intensity_weak_threshold
        mask_strong = positive_intensities < intensity_strong_threshold
        mask_plain = ~(mask_weak | mask_strong)
        
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
        
        params = {
            "hue_value": hue_value,
            "hue_width": hue_width,
            "saturation_minimum": saturation_minimum,
            "intensity_upper_limit": intensity_upper_limit,
            "intensity_weak_threshold": intensity_weak_threshold,
            "intensity_strong_threshold": intensity_strong_threshold,
            "intensity_lower_limit": intensity_lower_limit,
        }
        if extra_params:
            params.update(extra_params)
        
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
            "parameters": params,
            "metrics": {
                "execution_time_seconds": round(execution_time, 4),
                "cpu_percent": round(cpu_end.get("cpu_percent", 0.0), 1),
                "memory_mb": round(cpu_end.get("memory_mb", 0.0), 1),
            }
        }
    except Exception as e:
        logger.error(f"PPC HSI analysis failed for {item_id}: {e}", exc_info=True)
        raise


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
        # 
        # IMPORTANT: In HSI color space, "intensity" refers to pixel BRIGHTNESS (0=dark, 1=bright).
        # This is the opposite of "staining intensity"!
        #
        # For IHC staining (DAB/brown):
        #   - Strong staining = lots of DAB = dark brown = LOW pixel intensity (closer to 0)
        #   - Weak staining = little DAB = light brown = HIGH pixel intensity (closer to 1)
        #   - Negative = white background = VERY HIGH pixel intensity (close to 1)
        #
        # Examples:
        #   - Intensity < 0.05 = very dark pixels = strongest/darkest staining
        #   - Intensity < 0.3 = dark brown = strong staining
        #   - Intensity 0.3-0.6 = medium brown = plain staining
        #   - Intensity >= 0.6 = light brown = weak staining
        #   - Intensity > 0.9 = white/background = negative
        #
        # Classification:
        #   Weak: intensity >= intensity_weak_threshold (light brown, high pixel brightness)
        #   Strong: intensity < intensity_strong_threshold (dark brown, low pixel brightness)
        #   Plain: intensity_strong_threshold <= intensity < intensity_weak_threshold (medium brown)
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


def compute_ppc_region(
    item_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    output_width: int = 1024,
    method: str = "hsi",
    # HSI parameters
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
    Compute Positive Pixel Count (PPC) for a specific region of an image.
    
    Similar to compute_ppc but works on a cropped region instead of the full thumbnail.
    This allows analyzing specific areas at higher magnification or different FOV.
    
    The analysis logic is identical to thumbnail analysis - only the image fetch differs.
    
    Args:
        item_id: DSA item ID
        x: Left edge of region (0-1, normalized)
        y: Top edge of region (0-1, normalized)
        width: Width of region (0-1, normalized)
        height: Height of region (0-1, normalized)
        output_width: Output image width in pixels
        method: "hsi" (only HSI method supported now)
        hue_value: Center hue for positive color (0-1)
        hue_width: Width of hue range (0-1)
        saturation_minimum: Minimum saturation (0-1)
        intensity_upper_limit: Intensity above which pixel is negative (0-1)
        intensity_weak_threshold: Intensity threshold for weak vs plain (0-1)
        intensity_strong_threshold: Intensity threshold for plain vs strong (0-1)
        intensity_lower_limit: Intensity below which pixel is negative (0-1)
        dsa_client: Optional DSA client instance
    
    Returns:
        Dictionary with PPC results for the region
    """
    if dsa_client is None:
        dsa_client = get_dsa_client()
    
    # Get region image (only difference from thumbnail path)
    img = get_region_image(item_id, x, y, width, height, output_width, dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve region for item {item_id}")
    
    # Use common analysis function - identical code path to thumbnail analysis
    return _analyze_ppc_hsi(
        img=img,
        item_id=item_id,
        hue_value=hue_value,
        hue_width=hue_width,
        saturation_minimum=saturation_minimum,
        intensity_upper_limit=intensity_upper_limit,
        intensity_weak_threshold=intensity_weak_threshold,
        intensity_strong_threshold=intensity_strong_threshold,
        intensity_lower_limit=intensity_lower_limit,
        extra_params={
            "output_width": output_width,
            "region": {"x": x, "y": y, "width": width, "height": height}
        }
    )


#
# NOTE: get_positive_pixel_intensities is imported from app.services.ppc_label_images
# and re-exported via this module namespace for API stability.
