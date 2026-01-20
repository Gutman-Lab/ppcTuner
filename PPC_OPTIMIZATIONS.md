# PPC Algorithm Optimizations (ppcTunerV2)

This document summarizes the **performance**, **robustness**, and **accuracy** optimizations we implemented for the Positive Pixel Count (PPC) workflow in this repo, with notes on how to copy the changes into another Docker container/app.

## Goals / What changed at a high level

- **More accurate IHC detection**: Added/standardized an **HSI (HistomicsTK-style)** PPC path for DAB-like staining with hue wraparound handling and correct “dark = strong” intensity interpretation.
- **Much faster iteration**: Added **joblib** caching for expensive computations and for DSA thumbnail/region fetches.
- **More robust auto-tuning**: Implemented hue auto-detection that is biased toward “brown-ish” pixels and added a **multi-ROI sampling** endpoint to check stability across the slide.
- **Bounded ROI sampling runtime**: Sampled ROI logic screens candidates on the thumbnail first and enforces a hard `max_candidates` cap to avoid hangs.
- **Better debugging**: ROI sampling returns `debug_rois` explaining accepted/rejected candidates and why.

## Code map (backend)

Main entry points:

- `backend/app/api/ppc.py`
  - `/api/ppc/compute` (GET+POST)
  - `/api/ppc/compute-region`
  - `/api/ppc/auto-detect-hue`
  - `/api/ppc/auto-detect-hue-sampled` (**new**)
  - `/api/ppc/histogram`
  - `/api/ppc/label-image`, `/api/ppc/label-image-region`
  - `/api/ppc/intensity-map`
  - `/api/ppc/clear-cache`

Core services:

- `backend/app/services/ppc_service.py`: orchestrates PPC compute (thumbnail + region), caching, histogram + label image helpers
- `backend/app/services/ppc_thresholds.py`: auto-threshold + hue auto-detect + ROI sampling logic
- `backend/app/services/ppc_label_images.py`: label image generation + intensity-map extraction
- `backend/app/services/ppc_histogram.py`: histogram computation (cached, token-aware)
- `backend/app/services/image_fetching.py`: cached DSA thumbnail + region fetch
- `backend/app/services/ppc_cache.py`: shared joblib `Memory` (cache directory config)
- `backend/app/services/ppc_utils.py`: image hash for cache invalidation

Config:

- `backend/app/core/config.py`: `CACHE_DIR` (default `/app/.npCacheDir`), `DSA_BASE_URL`, `DSAKEY`

## Performance optimizations

### 1) Joblib caching (compute + I/O)

**What we did**

- Cached expensive, numpy-heavy functions with `joblib.Memory`:
  - PPC compute on thumbnails: `@memory.cache` on `_compute_ppc_cached` (RGB ratio) and `_compute_ppc_hsi_cached` (HSI)
  - Positive intensity extraction: `@memory.cache` on `_get_positive_pixel_intensities_cached`
  - Histogram computation: `@memory.cache` on `_compute_color_histogram_cached`
  - DSA image fetches (thumbnail + region): `@memory.cache` on `_fetch_thumbnail_uncached` and `_fetch_region_uncached`

**Why this matters**

- PPC tuning involves lots of repeated computations while you adjust parameters; caching makes the tool responsive.
- Region fetches are “expensive” network + decode + numpy conversion; caching avoids refetching the same ROI during exploration.

**Cache invalidation approach**

- We compute a short **image hash** from the thumbnail bytes and include it in compute cache keys:
  - `backend/app/services/ppc_utils.py::_get_image_hash(img)` → md5 of `img.tobytes()` (first 16 hex chars)
  - This is passed into cached functions as `image_hash`, so if the underlying slide thumbnail changes, cached results naturally miss.

**Auth-aware caching**

- Thumbnail/region fetch caches include the **DSA token** in the cache key to avoid mixing data across auth contexts:
  - `image_fetching.py`: `_fetch_thumbnail_uncached(..., token)` and `_fetch_region_uncached(..., token)`
  - `ppc_histogram.py`: `_compute_color_histogram_cached(..., token)`

### 2) Vectorized HSI PPC core (minimize copies)

In `ppc_service.py` `_analyze_ppc_hsi` / `_compute_ppc_hsi_cached`:

- Converts images to RGB (drop alpha via view `img[:, :, :3]`).
- Performs a single float32 normalization pass to \([0,1]\).
- Uses vectorized numpy masks and avoids repeatedly creating intermediate arrays.

#### Actual HSI PPC implementation (tile/thumbnail/region)

This is the **core PPC implementation** we currently use to compute PPC on an image array (thumbnail *or* fetched region). It is shared by the region endpoint so you can treat it as the “tile/ROI PPC core”:

```python
# backend/app/services/ppc_service.py
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
    start_time = time.time()
    cpu_start = _get_cpu_metrics()

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

    # Convert RGB to HSI (vectorized)
    hsi = rgb_to_hsi(img)
    h, s, i = hsi[..., 0], hsi[..., 1], hsi[..., 2]

    # Background mask (exclude very bright pixels)
    background_threshold = 0.94
    background_mask = (
        (img[..., 0] > background_threshold)
        & (img[..., 1] > background_threshold)
        & (img[..., 2] > background_threshold)
    )
    tissue_mask = ~background_mask

    # Apply tissue mask to HSI channels (indexing creates 1D views where possible)
    h_tissue = h[tissue_mask]
    s_tissue = s[tissue_mask]
    i_tissue = i[tissue_mask]

    # Hue range check with wraparound (HistomicsTK-style)
    hue_diff = ((h_tissue - hue_value + 0.5) % 1.0) - 0.5
    hue_in_range = np.abs(hue_diff) <= (hue_width / 2.0)

    # Saturation + intensity bounds
    saturation_ok = s_tissue >= saturation_minimum
    intensity_ok = (i_tissue < intensity_upper_limit) & (i_tissue >= intensity_lower_limit)

    # All positive pixels (meet all criteria)
    mask_all_positive = hue_in_range & saturation_ok & intensity_ok
    positive_intensities = i_tissue[mask_all_positive]

    # Classify positive pixels into weak/plain/strong based on intensity (brightness)
    mask_weak = positive_intensities >= intensity_weak_threshold
    mask_strong = positive_intensities < intensity_strong_threshold
    mask_plain = ~(mask_weak | mask_strong)

    weak_count = int(np.sum(mask_weak))
    plain_count = int(np.sum(mask_plain))
    strong_count = int(np.sum(mask_strong))
    total_positive = weak_count + plain_count + strong_count

    total_tissue_pixels = int(np.sum(tissue_mask))
    total_pixels = img.shape[0] * img.shape[1]

    weak_percentage = (weak_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
    plain_percentage = (plain_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
    strong_percentage = (strong_count / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0
    positive_percentage = (total_positive / total_tissue_pixels * 100) if total_tissue_pixels > 0 else 0.0

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
            "execution_time_seconds": round(time.time() - start_time, 4),
            "cpu_percent": round(cpu_end.get("cpu_percent", 0.0), 1),
            "memory_mb": round(cpu_end.get("memory_mb", 0.0), 1),
        },
    }
```

Notes on “fewer copies” in this implementation:

- `img[:, :, :3]` is a **view** (no full copy) when dropping alpha.
- We do **one** conversion to float32 + **one** `np.clip`.
- We slice tissue pixels once and do all classification in 1D arrays.

### 3) Bounded ROI sampling (no hangs)

In `ppc_thresholds.py::auto_detect_hue_parameters_sampled_rois`:

- **Screens candidate ROIs on the thumbnail** using cheap masks (tissue + DAB-band).
- Only fetches high-res regions for candidates likely to succeed.
- Enforces a hard cap: `max_candidates` (default `10`) to bound runtime.

## Algorithmic / robustness optimizations

### 1) Background/tissue masking consistently applied

Multiple modules use the same simple rule:

- Background pixels are those where **all RGB channels are very bright**.
- Threshold used in normalized space: `background_threshold = 0.94` (≈ 240/255).
- Tissue mask is `~background_mask`.

This keeps background from polluting histograms, hue detection, and PPC classification.

### 2) HSI PPC logic (HistomicsTK-style)

The HSI-based method is designed around DAB-like hue selection with wraparound-safe hue distance:

- **Hue wraparound**:
  - `hue_diff = ((h - hue_value + 0.5) % 1.0) - 0.5`
  - `abs(hue_diff) <= (hue_width/2)`

- **Positive pixel criteria**:
  - hue in range
  - saturation >= `saturation_minimum`
  - intensity between `[intensity_lower_limit, intensity_upper_limit)`

- **Intensity meaning** (critical for IHC):
  - In HSI, **intensity is brightness**.
  - Strong stain (dark brown) → **low intensity**.
  - Weak stain (light brown) → **high intensity**.

Classification:

- weak: `intensity >= intensity_weak_threshold`
- strong: `intensity < intensity_strong_threshold`
- plain: in-between

### 3) Improved hue auto-detect (thumbnail baseline)

In `ppc_thresholds.py::_auto_detect_hue_from_image`:

- Computes HSI on tissue pixels and picks candidate hue values.
- Biases toward “brown-ish” pixels via a simple RGB-based score:
  - `brown_score = (r - b) + 0.5*(r - g)`
  - Uses top ~30% of this score if enough pixels exist.
- Prefers expected DAB/brown hue band \(h \in [0, 0.25]\) when present.
- Finds hue peak via a histogram (180 bins) and estimates hue width via IQR.

### 4) Multi-ROI hue stability sampling (new endpoint)

Endpoint: `GET /api/ppc/auto-detect-hue-sampled`

Service: `ppc_thresholds.py::auto_detect_hue_parameters_sampled_rois`

**What it returns**

- `baseline`: thumbnail-level auto-detected hue params
- `rois`: accepted ROI measurements (each ROI: `x,y,width,height`, `hue_value`, `hue_width`, `tissue_fraction`, etc.)
- `debug_rois`: accepted + rejected candidates with reasons (for UI overlays)
- `summary`: aggregate stats (including circular hue statistics)

**Key parameters**

- `n_rois` (default 5): how many accepted ROIs to compute
- `roi_fraction` (default 0.08): normalized ROI square size
- `roi_output_width` (default 1024): high-res ROI fetch resolution
- `min_tissue_fraction` (default 0.6): reject ROIs with too much background
- `min_dab_band_fraction` (default 0.01): in DAB-biased mode, reject ROIs without enough “strong DAB-ish” pixels
- `sampling_mode`:
  - `dab_biased`: prefers DAB-band mask for center selection and enforces `min_dab_band_fraction`
  - `stratified`: spreads ROIs spatially across tissue; does **not** reject on DAB fraction (still reports it)
- `min_roi_separation`: prevents clustered ROIs (defaults to ~`roi_fraction * 0.9`)
- `max_candidates` (default 10): hard cap on candidates fully evaluated (bounded runtime)

**Weak/negative DAB handling**

- Computes `baseline_dab_band_fraction` from the thumbnail.
- If baseline DAB is extremely low (or mask is empty) and mode is `dab_biased`, the function enters a **likely-negative fast path**:
  - skips region fetches
  - returns thumbnail-only `debug_rois` with reasons so you can confirm “negative slide” vs “masking bug”

**Circular hue statistics**

- Hue values wrap around 0/1, so the summary includes:
  - `hue_value_circ_mean`
  - `hue_value_circ_std`

These are more robust than normal mean/std when values straddle the wrap point.

### 5) RGB ratio PPC improvements (more selective classes)

In `ppc_service.py::_compute_ppc_cached` (RGB ratio path):

- Uses a background/tissue mask first.
- Classifies brown/yellow/red with **more selective and mutually-exclusive** rules (not just simple thresholds).
- Uses “max blue” constraints and red/green balance to better approximate DAB-like brown.

#### Actual RGB-ratio classification excerpt

This is the core of the RGB-ratio PPC classifier (thumbnail path) showing the background mask + ratio logic + mutually exclusive class masks:

```python
# backend/app/services/ppc_service.py (excerpt from _compute_ppc_cached)
background_threshold = 0.94
background_mask = (r > background_threshold) & (g > background_threshold) & (b > background_threshold)
tissue_mask = ~background_mask

r_tissue = r[tissue_mask]
g_tissue = g[tissue_mask]
b_tissue = b[tissue_mask]

eps = 1e-6
total_intensity = r_tissue + g_tissue + b_tissue + eps
r_ratio = r_tissue / total_intensity
g_ratio = g_tissue / total_intensity
b_ratio = b_tissue / total_intensity

max_blue_brown = min(0.35, max(0.20, brown_threshold * 2.0))
brown_condition = (
    (b_ratio < max_blue_brown)
    & (r_ratio >= 0.30)
    & (g_ratio >= 0.30)
    & (r_ratio < 0.50)
    & (g_ratio < 0.50)
    & (np.abs(r_ratio - g_ratio) < 0.12)
    & (r_ratio + g_ratio > 0.65)
)

max_blue_yellow = min(0.25, max(0.10, 1.0 - yellow_threshold * 3.0))
yellow_condition = (
    ((r_ratio + g_ratio) >= yellow_threshold)
    & (b_ratio < max_blue_yellow)
    & (r_ratio >= 0.40)
    & (r_ratio > g_ratio * 1.1)
    & ~brown_condition
)

red_condition = (
    (r_ratio >= red_threshold)
    & (r_ratio > g_ratio * 1.3)
    & (r_ratio > b_ratio * 1.3)
    & ~brown_condition
    & ~yellow_condition
)
```

## Integration notes (copying into another Docker container)

### Minimal file set to copy

If you only want the PPC algorithm + sampling + caching:

- `backend/app/api/ppc.py`
- `backend/app/services/ppc_service.py`
- `backend/app/services/ppc_thresholds.py`
- `backend/app/services/ppc_label_images.py`
- `backend/app/services/ppc_histogram.py`
- `backend/app/services/ppc_cache.py`
- `backend/app/services/ppc_utils.py`
- `backend/app/services/image_fetching.py`
- `backend/app/services/dsa_client.py`
- `backend/app/core/config.py`

### Python dependencies

These are required by the backend PPC stack here (see `backend/requirements.txt`):

- `numpy`, `Pillow`, `requests`
- `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`
- `girder-client` (DSA/Girder API)
- `joblib` (cache)
- `psutil` (optional; metrics)

### Environment variables / config you need

At minimum:

- `DSA_BASE_URL` (e.g. `http://<dsa-host>:8080/api/v1`)
- `DSAKEY` (API key; backend exchanges it for a token via Girder)
- `CACHE_DIR` (default `/app/.npCacheDir`)

Make sure `CACHE_DIR` is writable inside your container (volume mount optional).

### API additions / changes to account for

If your other container needs the ROI sampling feature, ensure it exposes:

- `GET /api/ppc/auto-detect-hue-sampled`

And returns `debug_rois` + `summary` so the UI can draw accepted/rejected ROIs.

### Common “gotchas”

- **Token-aware caching**: if your auth token changes frequently, you’ll get more cache misses (expected). For stable performance, keep tokens relatively stable during a tuning session.
- **Cache directory**: joblib caches store potentially large artifacts; consider mounting `CACHE_DIR` to persistent storage if you want reuse across restarts.
- **DAB-band fraction semantics**: the DAB-band mask is intentionally a **strict / strong stain** heuristic. For weak-positive slides, `min_dab_band_fraction=0.01` is usually more realistic than `0.05`.

