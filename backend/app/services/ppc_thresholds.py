"""
Auto-threshold and auto-detect helpers for PPC tuning (RGB + HSI).
"""
import time
from typing import Any, Dict, Optional
import numpy as np
from app.services.dsa_client import DSAClient
from app.services.image_fetching import get_thumbnail_image, get_region_image, get_dsa_client
from app.services.color_analysis import rgb_to_hsi
from app.services.ppc_metrics import _get_cpu_metrics


def _auto_detect_hue_from_image(img: np.ndarray) -> Dict[str, float]:
    """
    Core hue auto-detection logic operating on an RGB image array.
    Expects img in either uint8 [0..255] or float [0..1].
    Returns the HSI parameters (no metrics).
    """
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
            "hue_value": 0.1,
            "hue_width": 0.1,
            "saturation_minimum": 0.1,
            "intensity_upper_limit": 0.9,
            "intensity_weak_threshold": 0.6,
            "intensity_strong_threshold": 0.3,
            "intensity_lower_limit": 0.05,
            "valid_pixel_count": 0.0,
            "dab_band_fraction": 0.0,
            "dab_band_count": 0.0,
        }

    valid_mask = (s_tissue > 0.1) & (i_tissue > 0.1) & (i_tissue < 0.9)
    h_valid = h_tissue[valid_mask]

    if len(h_valid) == 0:
        return {
            "hue_value": 0.1,
            "hue_width": 0.1,
            "saturation_minimum": 0.1,
            "intensity_upper_limit": 0.9,
            "intensity_weak_threshold": 0.6,
            "intensity_strong_threshold": 0.3,
            "intensity_lower_limit": 0.05,
            "valid_pixel_count": 0.0,
            "dab_band_fraction": 0.0,
            "dab_band_count": 0.0,
        }

    # Prefer "brown-ish" pixels when auto-detecting hue (bias away from hematoxylin blue).
    rgb_tissue = img[tissue_mask]
    rgb_valid = rgb_tissue[valid_mask]
    h_candidates = h_valid
    if rgb_valid.size > 0:
        r = rgb_valid[:, 0]
        g = rgb_valid[:, 1]
        b = rgb_valid[:, 2]
        brown_score = (r - b) + 0.5 * (r - g)
        score_thresh = np.percentile(brown_score, 70)
        brown_mask = brown_score >= score_thresh
        if np.count_nonzero(brown_mask) >= 250:
            h_candidates = h_valid[brown_mask]

    # Prefer expected DAB/brown hue band when present.
    hue_band_mask = (h_candidates >= 0.0) & (h_candidates <= 0.25)
    h_for_hist = h_candidates[hue_band_mask] if np.count_nonzero(hue_band_mask) >= 50 else h_candidates

    hist, bins = np.histogram(h_for_hist, bins=180)
    peak_idx = int(np.argmax(hist))
    hue_value = float((bins[peak_idx] + bins[peak_idx + 1]) / 2.0)

    q25, q75 = np.percentile(h_for_hist, [25, 75])
    hue_width = float((q75 - q25) * 2.0)
    hue_width = float(np.clip(hue_width, 0.05, 0.3))

    saturation_minimum = float(np.percentile(s_tissue[valid_mask], 10))
    saturation_minimum = float(np.clip(saturation_minimum, 0.05, 0.3))

    intensity_upper_limit = float(np.percentile(i_tissue, 95))
    intensity_lower_limit = float(np.percentile(i_tissue, 5))
    intensity_weak_threshold = float(np.percentile(i_tissue[valid_mask], 60))
    intensity_strong_threshold = float(np.percentile(i_tissue[valid_mask], 30))

    intensity_upper_limit = float(np.clip(intensity_upper_limit, 0.7, 0.98))
    intensity_lower_limit = float(np.clip(intensity_lower_limit, 0.02, 0.15))
    intensity_weak_threshold = float(np.clip(intensity_weak_threshold, 0.4, 0.8))
    intensity_strong_threshold = float(np.clip(intensity_strong_threshold, 0.15, 0.5))

    valid_pixel_count = int(len(h_valid))
    dab_band_count = int(np.count_nonzero((h_valid >= 0.0) & (h_valid <= 0.25)))
    dab_band_fraction = float(dab_band_count / max(1, valid_pixel_count))

    return {
        "hue_value": float(round(hue_value, 4)),
        "hue_width": float(round(hue_width, 4)),
        "saturation_minimum": float(round(saturation_minimum, 4)),
        "intensity_upper_limit": float(round(intensity_upper_limit, 4)),
        "intensity_weak_threshold": float(round(intensity_weak_threshold, 4)),
        "intensity_strong_threshold": float(round(intensity_strong_threshold, 4)),
        "intensity_lower_limit": float(round(intensity_lower_limit, 4)),
        "valid_pixel_count": float(valid_pixel_count),
        "dab_band_fraction": float(round(dab_band_fraction, 4)),
        "dab_band_count": float(dab_band_count),
    }


def auto_threshold_ppc(
    item_id: str,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Automatically determine optimal thresholds for RGB-based PPC computation.
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()

    if dsa_client is None:
        dsa_client = get_dsa_client()

    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")

    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    img = np.clip(img, 0.0, 1.0)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    background_threshold = 0.94
    background_mask = (r > background_threshold) & (g > background_threshold) & (b > background_threshold)
    tissue_mask = ~background_mask

    r_tissue = r[tissue_mask]
    g_tissue = g[tissue_mask]
    b_tissue = b[tissue_mask]

    if len(r_tissue) == 0:
        return {"brown_threshold": 0.15, "yellow_threshold": 0.20, "red_threshold": 0.30}

    eps = 1e-6
    total_intensity = r_tissue + g_tissue + b_tissue + eps
    r_ratio = r_tissue / total_intensity
    g_ratio = g_tissue / total_intensity
    b_ratio = b_tissue / total_intensity

    brown_metric = 1.0 - b_ratio
    brown_threshold = float(np.percentile(brown_metric, 75))

    yellow_metric = r_ratio + g_ratio
    yellow_threshold = float(np.percentile(yellow_metric, 70))

    red_threshold = float(np.percentile(r_ratio, 80))

    brown_threshold = float(np.clip(brown_threshold, 0.10, 0.50))
    yellow_threshold = float(np.clip(yellow_threshold, 0.15, 0.60))
    red_threshold = float(np.clip(red_threshold, 0.20, 0.70))

    end_time = time.time()
    cpu_end = _get_cpu_metrics()
    cpu_delta = {
        "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
        "memory_mb": cpu_end["memory_mb"],
    }

    return {
        "brown_threshold": round(brown_threshold, 3),
        "yellow_threshold": round(yellow_threshold, 3),
        "red_threshold": round(red_threshold, 3),
        "metrics": {
            "execution_time_seconds": round(end_time - start_time, 4),
            "cpu_percent": round(cpu_delta["cpu_percent"], 2),
            "memory_mb": round(cpu_delta["memory_mb"], 2),
        },
    }


def auto_detect_hue_parameters(
    item_id: str,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Automatically detect HSI hue parameters (biased toward DAB/brown).
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()

    if dsa_client is None:
        dsa_client = get_dsa_client()

    img = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if img is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")

    params = _auto_detect_hue_from_image(img)

    end_time = time.time()
    cpu_end = _get_cpu_metrics()
    cpu_delta = {
        "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
        "memory_mb": cpu_end["memory_mb"],
    }

    return {
        **params,
        "metrics": {
            "execution_time_seconds": round(end_time - start_time, 4),
            "cpu_percent": round(cpu_delta["cpu_percent"], 2),
            "memory_mb": round(cpu_delta["memory_mb"], 2),
        },
    }


def auto_detect_hue_parameters_sampled_rois(
    item_id: str,
    n_rois: int = 5,
    roi_fraction: float = 0.08,
    roi_output_width: int = 1024,
    min_tissue_fraction: float = 0.6,
    min_dab_band_fraction: float = 0.01,
    negative_dab_fraction_threshold: float = 0.005,
    min_roi_separation: Optional[float] = None,
    sampling_mode: str = "dab_biased",
    max_candidates: int = 10,
    thumbnail_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Dict[str, Any]:
    """
    Sample N square ROIs across tissue and compute auto-detected hue/hue_width for each ROI.

    Notes:
    - "roi_fraction" is normalized width/height (0..1) relative to full slide.
    - "roi_output_width" controls the pixel resolution fetched for each ROI (higher => more detail).
    """
    start_time = time.time()
    cpu_start = _get_cpu_metrics()

    if dsa_client is None:
        dsa_client = get_dsa_client()

    roi_fraction = float(np.clip(roi_fraction, 0.01, 0.5))
    n_rois = int(np.clip(n_rois, 1, 25))
    roi_output_width = int(np.clip(roi_output_width, 256, 4096))
    min_tissue_fraction = float(np.clip(min_tissue_fraction, 0.0, 0.99))
    min_dab_band_fraction = float(np.clip(min_dab_band_fraction, 0.0, 0.99))
    negative_dab_fraction_threshold = float(np.clip(negative_dab_fraction_threshold, 0.0, 0.5))
    max_candidates = int(np.clip(max_candidates, 1, 200))
    sampling_mode = str(sampling_mode or "dab_biased")
    if sampling_mode not in {"dab_biased", "stratified"}:
        sampling_mode = "dab_biased"
    # Default separation: roughly one ROI width (prevents heavy overlap).
    if min_roi_separation is None:
        min_roi_separation = float(roi_fraction * 0.9)
    min_roi_separation = float(np.clip(min_roi_separation, 0.0, 1.0))

    # Baseline from thumbnail
    thumb = get_thumbnail_image(item_id, width=thumbnail_width, dsa_client=dsa_client)
    if thumb is None:
        raise ValueError(f"Failed to retrieve thumbnail for item {item_id}")

    baseline = _auto_detect_hue_from_image(thumb)

    # Compute a simple tissue mask on the thumbnail to choose ROI centers.
    # We'll treat very bright pixels as background.
    if len(thumb.shape) == 2:
        thumb_rgb = np.stack([thumb, thumb, thumb], axis=-1)
    elif thumb.shape[2] == 4:
        thumb_rgb = thumb[:, :, :3]
    else:
        thumb_rgb = thumb

    if thumb_rgb.dtype == np.uint8:
        thumb_norm = thumb_rgb.astype(np.float32) / 255.0
    else:
        thumb_norm = thumb_rgb.astype(np.float32)
    thumb_norm = np.clip(thumb_norm, 0.0, 1.0)

    background_threshold = 0.94
    background_mask = (thumb_norm[..., 0] > background_threshold) & \
                      (thumb_norm[..., 1] > background_threshold) & \
                      (thumb_norm[..., 2] > background_threshold)
    tissue_mask = ~background_mask

    # Also compute a "DAB-band" mask on the thumbnail to avoid sampling hematoxylin-only tissue.
    # This is intentionally cheap and approximate: use HSI + basic bounds similar to auto-detect.
    hsi_thumb = rgb_to_hsi(thumb_norm)
    ht, st, it = hsi_thumb[..., 0], hsi_thumb[..., 1], hsi_thumb[..., 2]
    # Slightly permissive validity thresholds (helps with light brown / low-sat regions)
    valid_thumb_mask = tissue_mask & (st > 0.05) & (it > 0.05) & (it < 0.98)
    dab_band_mask = valid_thumb_mask & (ht >= 0.0) & (ht <= 0.25)

    Ht, Wt = int(thumb_norm.shape[0]), int(thumb_norm.shape[1])

    def _roi_mask_fraction(mask: np.ndarray, xn: float, yn: float, wn: float, hn: float) -> float:
        # Compute mask fraction of this ROI on the thumbnail mask.
        # Coordinates are normalized 0..1 relative to the full slide; we approximate via thumbnail.
        x0 = int(np.floor(xn * max(1, Wt - 1)))
        y0 = int(np.floor(yn * max(1, Ht - 1)))
        x1 = int(np.ceil((xn + wn) * max(1, Wt - 1)))
        y1 = int(np.ceil((yn + hn) * max(1, Ht - 1)))

        x0 = int(np.clip(x0, 0, max(0, Wt - 1)))
        y0 = int(np.clip(y0, 0, max(0, Ht - 1)))
        x1 = int(np.clip(x1, x0 + 1, Wt))
        y1 = int(np.clip(y1, y0 + 1, Ht))

        roi_mask = mask[y0:y1, x0:x1]
        if roi_mask.size == 0:
            return 0.0
        return float(np.mean(roi_mask))

    rng = np.random.default_rng()
    rois = []
    debug_rois = []

    ys, xs = np.where(tissue_mask)
    has_tissue_mask = bool(len(xs) > 0)
    ys_dab, xs_dab = np.where(dab_band_mask)
    has_dab_mask = bool(len(xs_dab) > 0)
    # If we can't find tissue at all on the thumbnail, fall back to sampling anywhere
    # and disable the min_tissue_fraction gate (otherwise we'd never accept any ROI).
    effective_min_tissue_fraction = min_tissue_fraction if has_tissue_mask else 0.0
    # If we can't find any DAB-band pixels on the thumbnail, disable that gate too.
    effective_min_dab_band_fraction = min_dab_band_fraction if has_dab_mask else 0.0
    # In stratified mode, we want spatial coverage across tissue even if DAB is sparse/absent.
    # So we don't reject candidates based on DAB fraction (we still *report* it for debugging).
    if sampling_mode == "stratified":
        effective_min_dab_band_fraction = 0.0

    baseline_dab_frac = float(baseline.get("dab_band_fraction", 0.0) or 0.0)
    likely_negative = (baseline_dab_frac < negative_dab_fraction_threshold) or (not has_dab_mask)

    # For likely-negative slides (little/no DAB), don't do expensive region fetches trying to satisfy DAB gating.
    # Instead, return quickly with debug candidates so we can confirm "negative" vs "masking bug".
    if likely_negative and min_dab_band_fraction > 0.0 and sampling_mode == "dab_biased":
        # Generate up to max_candidates thumbnail-only candidates and report why they'd be rejected.
        cand_attempts = 0
        while len(debug_rois) < max_candidates and cand_attempts < max_candidates * 50:
            cand_attempts += 1
            if has_tissue_mask and len(xs) > 0:
                idx = int(rng.integers(0, len(xs)))
                cx = xs[idx] / max(1, (Wt - 1))
                cy = ys[idx] / max(1, (Ht - 1))
            else:
                cx = float(rng.random())
                cy = float(rng.random())

            x = float(np.clip(cx - roi_fraction / 2.0, 0.0, 1.0 - roi_fraction))
            y = float(np.clip(cy - roi_fraction / 2.0, 0.0, 1.0 - roi_fraction))
            w = float(roi_fraction)
            h = float(roi_fraction)

            tissue_frac = _roi_mask_fraction(tissue_mask, x, y, w, h)
            dab_frac_thumb = _roi_mask_fraction(dab_band_mask, x, y, w, h)

            if tissue_frac < effective_min_tissue_fraction:
                reason = "tissue_fraction"
            else:
                reason = "likely_negative_no_dab"

            debug_rois.append({
                "status": "rejected",
                "reason": reason,
                "x": x, "y": y, "width": w, "height": h,
                "tissue_fraction": float(round(tissue_frac, 4)),
                "dab_band_fraction_thumb": float(round(dab_frac_thumb, 4)),
            })

        end_time = time.time()
        cpu_end = _get_cpu_metrics()
        cpu_delta = {
            "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
            "memory_mb": cpu_end["memory_mb"],
        }

        return {
            "item_id": item_id,
            "baseline": baseline,
            "rois": [],
            "debug_rois": debug_rois,
            "summary": {
                "n_rois_requested": n_rois,
                "n_rois_computed": 0,
                "roi_fraction": roi_fraction,
                "roi_output_width": roi_output_width,
                "min_tissue_fraction": effective_min_tissue_fraction,
                "min_dab_band_fraction": min_dab_band_fraction,
                "negative_dab_fraction_threshold": negative_dab_fraction_threshold,
                "sampling_mode": sampling_mode,
                "tissue_mask_found": has_tissue_mask,
                "dab_mask_found": has_dab_mask,
                "likely_negative": True,
                "baseline_dab_band_fraction": baseline_dab_frac,
                "attempts": int(cand_attempts),
                "max_candidates": int(max_candidates),
                "note": "Likely DAB-negative slide (or DAB mask empty). Skipped region fetches; returned thumbnail-only debug candidates.",
            },
            "metrics": {
                "execution_time_seconds": round(end_time - start_time, 4),
                "cpu_percent": round(cpu_delta["cpu_percent"], 2),
                "memory_mb": round(cpu_delta["memory_mb"], 2),
            },
        }

    # Keep sampling until we have enough ROIs (screening on thumbnail only),
    # then fetch exactly N regions (expensive) instead of potentially hundreds.
    max_attempts = int(max(200, n_rois * 400))
    attempts = 0
    rejected_tissue = 0
    rejected_dab = 0
    fetch_failures = 0
    candidates_evaluated = 0
    rejected_too_close = 0
    accepted_centers: list[tuple[float, float]] = []
    stratified_centers: list[tuple[float, float]] = []
    strat_idx = 0

    if sampling_mode == "stratified":
        # Build a grid of tissue-based candidate centers to encourage spatial coverage.
        grid_n = int(np.clip(int(np.ceil(np.sqrt(max_candidates))), 2, 16))
        for gy in range(grid_n):
            y0 = int(np.floor(gy * Ht / grid_n))
            y1 = int(np.floor((gy + 1) * Ht / grid_n))
            y1 = max(y0 + 1, min(Ht, y1))
            for gx in range(grid_n):
                x0 = int(np.floor(gx * Wt / grid_n))
                x1 = int(np.floor((gx + 1) * Wt / grid_n))
                x1 = max(x0 + 1, min(Wt, x1))
                cell = tissue_mask[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                pts = np.argwhere(cell)
                if pts.size == 0:
                    continue
                pick = pts[int(rng.integers(0, pts.shape[0]))]
                py = int(pick[0]) + y0
                px = int(pick[1]) + x0
                cxn = px / max(1, (Wt - 1))
                cyn = py / max(1, (Ht - 1))
                stratified_centers.append((float(cxn), float(cyn)))
        rng.shuffle(stratified_centers)

    while len(rois) < n_rois and attempts < max_attempts:
        # Hard cap for debugging: if we've evaluated enough candidates and still don't have enough
        # accepted ROIs, give up and return what we have.
        if candidates_evaluated >= max_candidates:
            break

        attempts += 1

        if sampling_mode == "stratified" and strat_idx < len(stratified_centers):
            cx, cy = stratified_centers[strat_idx]
            strat_idx += 1
        else:
            # Prefer sampling centers from DAB-band pixels when available (more stable for aBeta/DAB slides),
            # otherwise fall back to any tissue, then to anywhere.
            if has_dab_mask and len(xs_dab) > 0:
                idx = int(rng.integers(0, len(xs_dab)))
                cx = xs_dab[idx] / max(1, (Wt - 1))
                cy = ys_dab[idx] / max(1, (Ht - 1))
            elif has_tissue_mask and len(xs) > 0:
                idx = int(rng.integers(0, len(xs)))
                cx = xs[idx] / max(1, (Wt - 1))
                cy = ys[idx] / max(1, (Ht - 1))
            else:
                cx = float(rng.random())
                cy = float(rng.random())

        x = float(np.clip(cx - roi_fraction / 2.0, 0.0, 1.0 - roi_fraction))
        y = float(np.clip(cy - roi_fraction / 2.0, 0.0, 1.0 - roi_fraction))
        w = float(roi_fraction)
        h = float(roi_fraction)

        tissue_frac = _roi_mask_fraction(tissue_mask, x, y, w, h)
        if tissue_frac < effective_min_tissue_fraction:
            rejected_tissue += 1
            if len(debug_rois) < max_candidates:
                debug_rois.append({
                    "status": "rejected",
                    "reason": "tissue_fraction",
                    "x": x, "y": y, "width": w, "height": h,
                    "tissue_fraction": float(round(tissue_frac, 4)),
                    "dab_band_fraction_thumb": None,
                })
            continue

        dab_frac_thumb = _roi_mask_fraction(dab_band_mask, x, y, w, h)
        if dab_frac_thumb < effective_min_dab_band_fraction:
            rejected_dab += 1
            if len(debug_rois) < max_candidates:
                debug_rois.append({
                    "status": "rejected",
                    "reason": "dab_band_fraction_thumb",
                    "x": x, "y": y, "width": w, "height": h,
                    "tissue_fraction": float(round(tissue_frac, 4)),
                    "dab_band_fraction_thumb": float(round(dab_frac_thumb, 4)),
                })
            continue

        candidates_evaluated += 1

        # Enforce spatial separation between accepted ROIs to avoid clustered/overlapping samples.
        cxn = x + w / 2.0
        cyn = y + h / 2.0
        too_close = False
        for (px, py) in accepted_centers:
            if ((cxn - px) ** 2 + (cyn - py) ** 2) < (min_roi_separation ** 2):
                too_close = True
                break
        if too_close:
            rejected_too_close += 1
            if len(debug_rois) < max_candidates:
                debug_rois.append({
                    "status": "rejected",
                    "reason": "too_close",
                    "x": x, "y": y, "width": w, "height": h,
                    "tissue_fraction": float(round(tissue_frac, 4)),
                    "dab_band_fraction_thumb": float(round(dab_frac_thumb, 4)),
                })
            continue

        region_img = get_region_image(item_id, x, y, w, h, roi_output_width, dsa_client)
        if region_img is None:
            fetch_failures += 1
            if len(debug_rois) < max_candidates:
                debug_rois.append({
                    "status": "rejected",
                    "reason": "region_fetch_failed",
                    "x": x, "y": y, "width": w, "height": h,
                    "tissue_fraction": float(round(tissue_frac, 4)),
                    "dab_band_fraction_thumb": float(round(dab_frac_thumb, 4)),
                })
            continue

        roi_params = _auto_detect_hue_from_image(region_img)
        roi_entry = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "output_width": roi_output_width,
            "tissue_fraction": float(round(tissue_frac, 4)),
            "hue_value": roi_params["hue_value"],
            "hue_width": roi_params["hue_width"],
            "dab_band_fraction": float(roi_params.get("dab_band_fraction", 0.0)),
            "dab_band_fraction_thumb": float(round(dab_frac_thumb, 4)),
        }
        rois.append(roi_entry)
        accepted_centers.append((cxn, cyn))
        if len(debug_rois) < max_candidates:
            debug_rois.append({
                "status": "accepted",
                "reason": None,
                **roi_entry,
            })

    hue_values = np.array([r["hue_value"] for r in rois], dtype=np.float32) if rois else np.array([], dtype=np.float32)
    hue_widths = np.array([r["hue_width"] for r in rois], dtype=np.float32) if rois else np.array([], dtype=np.float32)

    # Circular stats for hue (robust when values wrap around 0/1).
    if hue_values.size:
        angles = hue_values.astype(np.float64) * 2.0 * np.pi
        sin_m = float(np.mean(np.sin(angles)))
        cos_m = float(np.mean(np.cos(angles)))
        mean_angle = float(np.arctan2(sin_m, cos_m))
        if mean_angle < 0:
            mean_angle += 2.0 * np.pi
        hue_circ_mean = float(mean_angle / (2.0 * np.pi))
        R = float(np.hypot(sin_m, cos_m))
        hue_circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(1e-12, R)))) / (2.0 * np.pi))
    else:
        hue_circ_mean = None
        hue_circ_std = None

    summary = {
        "n_rois_requested": n_rois,
        "n_rois_computed": int(len(rois)),
        "roi_fraction": roi_fraction,
        "roi_output_width": roi_output_width,
        "min_tissue_fraction": effective_min_tissue_fraction,
        "min_dab_band_fraction": effective_min_dab_band_fraction,
        "max_candidates": int(max_candidates),
        "min_roi_separation": float(min_roi_separation),
        "sampling_mode": sampling_mode,
        "tissue_mask_found": has_tissue_mask,
        "dab_mask_found": has_dab_mask,
        "attempts": int(attempts),
        "candidates_evaluated": int(candidates_evaluated),
        "rejected_tissue": int(rejected_tissue),
        "rejected_dab": int(rejected_dab),
        "rejected_too_close": int(rejected_too_close),
        "fetch_failures": int(fetch_failures),
        "hue_value_mean": float(np.mean(hue_values)) if hue_values.size else None,
        "hue_value_std": float(np.std(hue_values)) if hue_values.size else None,
        "hue_value_circ_mean": hue_circ_mean,
        "hue_value_circ_std": hue_circ_std,
        "hue_width_mean": float(np.mean(hue_widths)) if hue_widths.size else None,
        "hue_width_std": float(np.std(hue_widths)) if hue_widths.size else None,
        "baseline_hue_value": baseline["hue_value"],
        "baseline_hue_width": baseline["hue_width"],
        "likely_negative": bool(likely_negative),
        "baseline_dab_band_fraction": baseline_dab_frac,
        "negative_dab_fraction_threshold": negative_dab_fraction_threshold,
    }

    end_time = time.time()
    cpu_end = _get_cpu_metrics()
    cpu_delta = {
        "cpu_percent": max(0.0, cpu_end["cpu_percent"] - cpu_start["cpu_percent"]),
        "memory_mb": cpu_end["memory_mb"],
    }

    return {
        "item_id": item_id,
        "baseline": baseline,
        "rois": rois,
        "debug_rois": debug_rois,
        "summary": summary,
        "metrics": {
            "execution_time_seconds": round(end_time - start_time, 4),
            "cpu_percent": round(cpu_delta["cpu_percent"], 2),
            "memory_mb": round(cpu_delta["memory_mb"], 2),
        },
    }

