"""
Label image + intensity helpers for PPC overlays (cached where appropriate).
"""
import time
from typing import Any, Dict, Optional
import numpy as np
import logging
from app.services.dsa_client import DSAClient
from app.services.image_fetching import get_thumbnail_image, get_region_image, get_dsa_client
from app.services.color_analysis import rgb_to_hsi
from app.services.ppc_cache import memory
from app.services.ppc_utils import _get_image_hash

logger = logging.getLogger(__name__)


def get_ppc_label_image(
    item_id: str,
    method: str = "hsi",
    thumbnail_width: int = 1024,
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_weak_threshold: float = 0.6,
    intensity_strong_threshold: float = 0.3,
    intensity_lower_limit: float = 0.05,
    dsa_client: Optional[DSAClient] = None
) -> Optional[np.ndarray]:
    """Generate label image for HSI-based PPC visualization on a thumbnail."""
    if method != "hsi":
        return None

    if dsa_client is None:
        dsa_client = get_dsa_client()

    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    hsi = rgb_to_hsi(img)
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    background_threshold = 0.94
    background_mask = (img[..., 0] > background_threshold) & \
                      (img[..., 1] > background_threshold) & \
                      (img[..., 2] > background_threshold)
    tissue_mask = ~background_mask

    label_image = np.zeros(img.shape[:2], dtype=np.uint8)

    h_tissue = h[tissue_mask]
    s_tissue = s[tissue_mask]
    i_tissue = i[tissue_mask]
    if len(h_tissue) == 0:
        return label_image

    hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
    hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
    saturation_ok = s_tissue >= saturation_minimum
    intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
    mask_all_positive = hue_in_range & saturation_ok & intensity_ok

    mask_all_positive_full = np.zeros(img.shape[:2], dtype=bool)
    mask_all_positive_full[tissue_mask] = mask_all_positive

    if np.any(mask_all_positive_full):
        positive_intensities_full = i[mask_all_positive_full]
        mask_weak_full = positive_intensities_full >= intensity_weak_threshold
        mask_strong_full = positive_intensities_full < intensity_strong_threshold
        mask_plain_full = ~(mask_weak_full | mask_strong_full)

        positive_indices = np.where(mask_all_positive_full)
        weak_indices = (positive_indices[0][mask_weak_full], positive_indices[1][mask_weak_full])
        plain_indices = (positive_indices[0][mask_plain_full], positive_indices[1][mask_plain_full])
        strong_indices = (positive_indices[0][mask_strong_full], positive_indices[1][mask_strong_full])

        label_image[weak_indices] = 1
        label_image[plain_indices] = 2
        label_image[strong_indices] = 3

    return label_image


def get_ppc_label_image_region(
    item_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    output_width: int = 1024,
    method: str = "hsi",
    hue_value: float = 0.1,
    hue_width: float = 0.1,
    saturation_minimum: float = 0.1,
    intensity_upper_limit: float = 0.9,
    intensity_weak_threshold: float = 0.6,
    intensity_strong_threshold: float = 0.3,
    intensity_lower_limit: float = 0.05,
    dsa_client: Optional[DSAClient] = None
) -> Optional[np.ndarray]:
    """Generate label image for HSI-based PPC visualization on a region."""
    if method != "hsi":
        return None

    if dsa_client is None:
        dsa_client = get_dsa_client()

    img = get_region_image(item_id, x, y, width, height, output_width, dsa_client)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        img = (img.astype(np.float32) / 255.0)
    else:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    hsi = rgb_to_hsi(img)
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    background_threshold = 0.94
    background_mask = (img[..., 0] > background_threshold) & \
                      (img[..., 1] > background_threshold) & \
                      (img[..., 2] > background_threshold)
    tissue_mask = ~background_mask

    h_tissue = h[tissue_mask]
    s_tissue = s[tissue_mask]
    i_tissue = i[tissue_mask]
    if len(h_tissue) == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
    hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
    saturation_ok = s_tissue >= saturation_minimum
    intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
    mask_all_positive = hue_in_range & saturation_ok & intensity_ok

    positive_intensities = i_tissue[mask_all_positive]
    if len(positive_intensities) == 0:
        return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    mask_weak = positive_intensities >= intensity_weak_threshold
    mask_strong = positive_intensities < intensity_strong_threshold
    mask_plain = ~(mask_weak | mask_strong)

    label_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    tissue_indices = np.where(tissue_mask)
    positive_tissue_indices = np.where(mask_all_positive)[0]

    weak_indices = positive_tissue_indices[mask_weak]
    plain_indices = positive_tissue_indices[mask_plain]
    strong_indices = positive_tissue_indices[mask_strong]

    for idx in weak_indices:
        label_image[tissue_indices[0][idx], tissue_indices[1][idx]] = 1
    for idx in plain_indices:
        label_image[tissue_indices[0][idx], tissue_indices[1][idx]] = 2
    for idx in strong_indices:
        label_image[tissue_indices[0][idx], tissue_indices[1][idx]] = 3

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
    """Cached helper for positive pixel intensities (thresholds excluded)."""
    start_time = time.time()
    try:
        dsa_client = get_dsa_client()
        img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
        if img is None:
            raise ValueError("Failed to retrieve thumbnail image")

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype == np.uint8:
            img = (img.astype(np.float32) / 255.0)
        else:
            img = img.astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        hsi = rgb_to_hsi(img)
        h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]

        background_threshold = 0.94
        background_mask = (img[..., 0] > background_threshold) & \
                          (img[..., 1] > background_threshold) & \
                          (img[..., 2] > background_threshold)
        tissue_mask = ~background_mask

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
                "metrics": {"execution_time_seconds": round(time.time() - start_time, 4)},
            }

        hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
        hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)
        saturation_ok = s_tissue >= saturation_minimum
        intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)
        mask_all_positive = hue_in_range & saturation_ok & intensity_ok

        positive_intensities = i_tissue[mask_all_positive].tolist()

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
            "metrics": {"execution_time_seconds": round(time.time() - start_time, 4)},
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
    """Public wrapper for cached positive pixel intensities."""
    if dsa_client is None:
        dsa_client = get_dsa_client()

    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")
    img_hash = _get_image_hash(img)

    return _get_positive_pixel_intensities_cached(
        item_id,
        img_hash,
        hue_value,
        hue_width,
        saturation_minimum,
        intensity_upper_limit,
        intensity_lower_limit,
        thumbnail_width,
    )

