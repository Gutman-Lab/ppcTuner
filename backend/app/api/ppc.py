"""PPC (Positive Pixel Count) endpoints"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import logging
from app.services.ppc_service import compute_ppc, compute_ppc_region, auto_threshold_ppc, auto_detect_hue_parameters, get_ppc_label_image, get_ppc_label_image_region, get_positive_pixel_intensities, memory
from fastapi.responses import Response
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class PPCRequest(BaseModel):
    """PPC computation request model"""
    item_id: str
    method: str = "rgb_ratio"  # "rgb_ratio" or "hsi"
    # RGB method parameters
    brown_threshold: float = 0.15
    yellow_threshold: float = 0.20
    red_threshold: float = 0.30
    # HSI method parameters
    hue_value: float = 0.1
    hue_width: float = 0.1
    saturation_minimum: float = 0.1
    intensity_upper_limit: float = 0.9
    intensity_weak_threshold: float = 0.6
    intensity_strong_threshold: float = 0.3
    intensity_lower_limit: float = 0.05
    thumbnail_width: int = 1024


class PPCResponse(BaseModel):
    """PPC computation response model (flexible for both RGB and HSI methods)"""
    item_id: str
    total_pixels: int
    tissue_pixels: int
    background_pixels: int
    method: str  # "rgb_ratio" or "hsi"
    # RGB method results (optional)
    brown_pixels: Optional[int] = None
    yellow_pixels: Optional[int] = None
    red_pixels: Optional[int] = None
    brown_percentage: Optional[float] = None
    yellow_percentage: Optional[float] = None
    red_percentage: Optional[float] = None
    # HSI method results (optional)
    weak_positive_pixels: Optional[int] = None
    plain_positive_pixels: Optional[int] = None
    strong_positive_pixels: Optional[int] = None
    weak_percentage: Optional[float] = None
    plain_percentage: Optional[float] = None
    strong_percentage: Optional[float] = None
    # Common results
    total_positive_pixels: int
    positive_percentage: float
    parameters: dict
    metrics: dict  # Timing and CPU metrics


@router.post("/compute", response_model=PPCResponse)
async def compute_ppc_endpoint(request: PPCRequest):
    """
    Compute Positive Pixel Count (PPC) for an image.
    
    Supports two methods:
    - "rgb_ratio": Simple RGB ratio-based classification (faster)
    - "hsi": HSI color space-based classification (HistomicsTK-style, more accurate)
    
    Results are cached based on image hash and parameters.
    """
    try:
        result = compute_ppc(
            item_id=request.item_id,
            method=request.method,
            brown_threshold=request.brown_threshold,
            yellow_threshold=request.yellow_threshold,
            red_threshold=request.red_threshold,
            thumbnail_width=request.thumbnail_width,
            # HSI parameters
            hue_value=request.hue_value,
            hue_width=request.hue_width,
            saturation_minimum=request.saturation_minimum,
            intensity_upper_limit=request.intensity_upper_limit,
            intensity_weak_threshold=request.intensity_weak_threshold,
            intensity_strong_threshold=request.intensity_strong_threshold,
            intensity_lower_limit=request.intensity_lower_limit,
        )
        return PPCResponse(**result)
    except ValueError as e:
        logger.error(f"PPC computation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PPC computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PPC computation failed: {str(e)}")


@router.get("/compute", response_model=PPCResponse)
async def compute_ppc_get(
    item_id: str = Query(..., description="DSA item ID"),
    method: str = Query("rgb_ratio", description="PPC method: 'rgb_ratio' or 'hsi'"),
    # RGB method parameters
    brown_threshold: float = Query(0.15, ge=0.0, le=1.0, description="Brown pixel threshold (RGB method)"),
    yellow_threshold: float = Query(0.20, ge=0.0, le=1.0, description="Yellow pixel threshold (RGB method)"),
    red_threshold: float = Query(0.30, ge=0.0, le=1.0, description="Red pixel threshold (RGB method)"),
    # HSI method parameters
    hue_value: float = Query(0.1, ge=0.0, le=1.0, description="Center hue for positive color (HSI method)"),
    hue_width: float = Query(0.1, ge=0.0, le=1.0, description="Width of hue range (HSI method)"),
    saturation_minimum: float = Query(0.1, ge=0.0, le=1.0, description="Minimum saturation (HSI method)"),
    intensity_upper_limit: float = Query(0.9, ge=0.0, le=1.0, description="Intensity upper limit (HSI method)"),
    intensity_weak_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Intensity weak threshold (HSI method)"),
    intensity_strong_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Intensity strong threshold (HSI method)"),
    intensity_lower_limit: float = Query(0.05, ge=0.0, le=1.0, description="Intensity lower limit (HSI method)"),
    thumbnail_width: int = Query(1024, ge=256, le=4096, description="Thumbnail width")
):
    """
    Compute Positive Pixel Count (PPC) for an image (GET endpoint).
    
    Supports two methods:
    - "rgb_ratio": Simple RGB ratio-based classification (faster)
    - "hsi": HSI color space-based classification (HistomicsTK-style, more accurate)
    
    Results are cached based on image hash and parameters.
    """
    try:
        result = compute_ppc(
            item_id=item_id,
            method=method,
            brown_threshold=brown_threshold,
            yellow_threshold=yellow_threshold,
            red_threshold=red_threshold,
            thumbnail_width=thumbnail_width,
            # HSI parameters
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_weak_threshold=intensity_weak_threshold,
            intensity_strong_threshold=intensity_strong_threshold,
            intensity_lower_limit=intensity_lower_limit,
        )
        return PPCResponse(**result)
    except ValueError as e:
        logger.error(f"PPC computation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PPC computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PPC computation failed: {str(e)}")


@router.get("/histogram")
async def get_color_histogram(
    item_id: str = Query(..., description="DSA item ID"),
    width: int = Query(1024, ge=256, le=4096, description="Thumbnail width"),
    bins: int = Query(256, ge=32, le=512, description="Number of histogram bins")
):
    """
    Compute color histogram for an image thumbnail.
    
    Returns RGB histograms and statistics (mean, std) for each channel.
    Results are cached based on image hash.
    """
    try:
        from app.services.ppc_service import compute_color_histogram
        result = compute_color_histogram(
            item_id=item_id,
            width=width,
            bins=bins
        )
        return result
    except ValueError as e:
        logger.error(f"Histogram computation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Histogram computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Histogram computation failed: {str(e)}")


@router.get("/auto-threshold")
async def get_auto_threshold(
    item_id: str = Query(..., description="DSA item ID"),
    thumbnail_width: int = Query(1024, ge=256, le=4096, description="Thumbnail width")
):
    """
    Automatically determine optimal thresholds for PPC computation.
    
    Uses histogram analysis to suggest brown, yellow, and red thresholds
    based on the image characteristics.
    """
    try:
        result = auto_threshold_ppc(
            item_id=item_id,
            thumbnail_width=thumbnail_width
        )
        return result
    except ValueError as e:
        logger.error(f"Auto-threshold error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Auto-threshold failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Auto-threshold failed: {str(e)}")


@router.get("/auto-detect-hue")
async def auto_detect_hue(
    item_id: str = Query(..., description="DSA item ID"),
    thumbnail_width: int = Query(1024, ge=256, le=4096, description="Thumbnail width")
):
    """
    Automatically detect optimal hue_value and hue_width for HSI-based PPC.
    
    Analyzes the image to find the dominant brown/DAB hue and suggests parameters.
    """
    try:
        result = auto_detect_hue_parameters(
            item_id=item_id,
            thumbnail_width=thumbnail_width
        )
        return result
    except ValueError as e:
        logger.error(f"Auto-detect hue error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Auto-detect hue failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Auto-detect hue failed: {str(e)}")


@router.get("/label-image", response_class=Response)
async def get_label_image(
    item_id: str = Query(..., description="DSA item ID"),
    method: str = Query("hsi", description="PPC method (only 'hsi' supported for label images)"),
    thumbnail_width: int = Query(1024, ge=256, le=4096, description="Thumbnail width"),
    # HSI parameters
    hue_value: float = Query(0.1, ge=0.0, le=1.0, description="Center hue (HSI method)"),
    hue_width: float = Query(0.1, ge=0.0, le=1.0, description="Hue width (HSI method)"),
    saturation_minimum: float = Query(0.1, ge=0.0, le=1.0, description="Min saturation (HSI method)"),
    intensity_upper_limit: float = Query(0.9, ge=0.0, le=1.0, description="Intensity upper limit (HSI method)"),
    intensity_weak_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Intensity weak threshold (HSI method)"),
    intensity_strong_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Intensity strong threshold (HSI method)"),
    intensity_lower_limit: float = Query(0.05, ge=0.0, le=1.0, description="Intensity lower limit (HSI method)"),
    # Display options
    show_weak: bool = Query(True, description="Show weak positive pixels"),
    show_plain: bool = Query(True, description="Show plain positive pixels"),
    show_strong: bool = Query(True, description="Show strong positive pixels"),
    color_scheme: str = Query("blue-green-red", description="Color scheme: 'blue-green-red' or 'yellow-orange-red'")
):
    """
    Get label image for PPC visualization.
    
    Returns a PNG image where pixel colors represent classification:
    - Black (0) = negative/background
    - Blue (1) = weak positive
    - Green (2) = plain positive
    - Red (3) = strong positive
    
    Only HSI method currently supports label images.
    """
    try:
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        label_image = get_ppc_label_image(
            item_id=item_id,
            method=method,
            thumbnail_width=thumbnail_width,
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_weak_threshold=intensity_weak_threshold,
            intensity_strong_threshold=intensity_strong_threshold,
            intensity_lower_limit=intensity_lower_limit,
        )
        
        if label_image is None:
            raise HTTPException(status_code=400, detail="Label images only supported for HSI method")
        
        # Convert label image to colored visualization
        height, width = label_image.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Apply color scheme
        if color_scheme == "yellow-orange-red":
            # Yellow for weak (1), Orange for plain (2), Red for strong (3)
            if show_weak:
                colored[label_image == 1] = [255, 255, 0]  # Yellow
            if show_plain:
                colored[label_image == 2] = [255, 165, 0]  # Orange
            if show_strong:
                colored[label_image == 3] = [255, 0, 0]  # Red
        else:  # blue-green-red (default)
            # Blue for weak (1), Green for plain (2), Red for strong (3)
            if show_weak:
                colored[label_image == 1] = [0, 0, 255]  # Blue
            if show_plain:
                colored[label_image == 2] = [0, 255, 0]  # Green
            if show_strong:
                colored[label_image == 3] = [255, 0, 0]  # Red
        
        # Convert to PIL Image and then to PNG bytes
        img = Image.fromarray(colored)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
            }
        )
    except ValueError as e:
        logger.error(f"Label image error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Label image generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Label image generation failed: {str(e)}")


@router.get("/compute-region", response_model=PPCResponse)
async def compute_ppc_region_endpoint(
    item_id: str = Query(..., description="DSA item ID"),
    x: float = Query(..., ge=0.0, le=1.0, description="Left edge of region (0-1, normalized)"),
    y: float = Query(..., ge=0.0, le=1.0, description="Top edge of region (0-1, normalized)"),
    width: float = Query(..., ge=0.0, le=1.0, description="Width of region (0-1, normalized)"),
    height: float = Query(..., ge=0.0, le=1.0, description="Height of region (0-1, normalized)"),
    output_width: int = Query(1024, ge=256, le=4096, description="Output image width in pixels"),
    # HSI parameters
    hue_value: float = Query(0.1, ge=0.0, le=1.0, description="Center hue for positive color (HSI method)"),
    hue_width: float = Query(0.1, ge=0.0, le=1.0, description="Width of hue range (HSI method)"),
    saturation_minimum: float = Query(0.1, ge=0.0, le=1.0, description="Minimum saturation (HSI method)"),
    intensity_upper_limit: float = Query(0.9, ge=0.0, le=1.0, description="Intensity upper limit (HSI method)"),
    intensity_weak_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Intensity weak threshold (HSI method)"),
    intensity_strong_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Intensity strong threshold (HSI method)"),
    intensity_lower_limit: float = Query(0.05, ge=0.0, le=1.0, description="Intensity lower limit (HSI method)")
):
    """
    Compute Positive Pixel Count (PPC) for a specific region of an image.
    
    Similar to /compute but works on a cropped region instead of the full thumbnail.
    This allows analyzing specific areas at higher magnification or different FOV.
    The region is fetched from DSA at the requested resolution.
    
    Results are computed on the region image (similar format to thumbnail).
    """
    try:
        result = compute_ppc_region(
            item_id=item_id,
            x=x,
            y=y,
            width=width,
            height=height,
            output_width=output_width,
            method="hsi",
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_weak_threshold=intensity_weak_threshold,
            intensity_strong_threshold=intensity_strong_threshold,
            intensity_lower_limit=intensity_lower_limit,
        )
        return PPCResponse(**result)
    except ValueError as e:
        logger.error(f"PPC region computation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"PPC region computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PPC region computation failed: {str(e)}")


@router.get("/label-image-region", response_class=Response)
async def get_label_image_region(
    item_id: str = Query(..., description="DSA item ID"),
    x: float = Query(..., ge=0.0, le=1.0, description="Left edge of region (0-1, normalized)"),
    y: float = Query(..., ge=0.0, le=1.0, description="Top edge of region (0-1, normalized)"),
    width: float = Query(..., ge=0.0, le=1.0, description="Width of region (0-1, normalized)"),
    height: float = Query(..., ge=0.0, le=1.0, description="Height of region (0-1, normalized)"),
    output_width: int = Query(1024, ge=256, le=4096, description="Output image width in pixels"),
    method: str = Query("hsi", description="PPC method (only 'hsi' supported for label images)"),
    # HSI parameters
    hue_value: float = Query(0.1, ge=0.0, le=1.0, description="Center hue (HSI method)"),
    hue_width: float = Query(0.1, ge=0.0, le=1.0, description="Hue width (HSI method)"),
    saturation_minimum: float = Query(0.1, ge=0.0, le=1.0, description="Min saturation (HSI method)"),
    intensity_upper_limit: float = Query(0.9, ge=0.0, le=1.0, description="Intensity upper limit (HSI method)"),
    intensity_weak_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Intensity weak threshold (HSI method)"),
    intensity_strong_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Intensity strong threshold (HSI method)"),
    intensity_lower_limit: float = Query(0.05, ge=0.0, le=1.0, description="Intensity lower limit (HSI method)"),
    show_weak: bool = Query(True, description="Show weak positive pixels"),
    show_plain: bool = Query(True, description="Show plain positive pixels"),
    show_strong: bool = Query(True, description="Show strong positive pixels"),
    color_scheme: str = Query("blue-green-red", description="Color scheme: 'blue-green-red' or 'yellow-orange-red'")
):
    """
    Get PPC label image for a specific region of an image.
    
    Similar to /label-image but works on a cropped region instead of the full thumbnail.
    """
    try:
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        label_image = get_ppc_label_image_region(
            item_id=item_id,
            x=x,
            y=y,
            width=width,
            height=height,
            output_width=output_width,
            method=method,
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_weak_threshold=intensity_weak_threshold,
            intensity_strong_threshold=intensity_strong_threshold,
            intensity_lower_limit=intensity_lower_limit,
        )
        
        if label_image is None:
            raise HTTPException(status_code=400, detail="Label images only supported for HSI method")
        
        # Convert label image to colored visualization
        height_img, width_img = label_image.shape
        colored = np.zeros((height_img, width_img, 3), dtype=np.uint8)
        
        # Apply color scheme
        if color_scheme == "yellow-orange-red":
            # Yellow for weak (1), Orange for plain (2), Red for strong (3)
            if show_weak:
                colored[label_image == 1] = [255, 255, 0]  # Yellow
            if show_plain:
                colored[label_image == 2] = [255, 165, 0]  # Orange
            if show_strong:
                colored[label_image == 3] = [255, 0, 0]  # Red
        else:  # blue-green-red (default)
            # Blue for weak (1), Green for plain (2), Red for strong (3)
            if show_weak:
                colored[label_image == 1] = [0, 0, 255]  # Blue
            if show_plain:
                colored[label_image == 2] = [0, 255, 0]  # Green
            if show_strong:
                colored[label_image == 3] = [255, 0, 0]  # Red
        
        # Convert to PIL Image and then to PNG bytes
        img = Image.fromarray(colored)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return Response(
            content=img_bytes.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600",
            }
        )
    except ValueError as e:
        logger.error(f"Label image region error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Label image region generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Label image region generation failed: {str(e)}")


@router.get("/intensity-map")
async def get_intensity_map(
    item_id: str = Query(..., description="DSA item ID"),
    thumbnail_width: int = Query(1024, ge=256, le=4096, description="Thumbnail width"),
    # HSI parameters (excluding thresholds)
    hue_value: float = Query(0.1, ge=0.0, le=1.0, description="Center hue"),
    hue_width: float = Query(0.1, ge=0.0, le=1.0, description="Hue width"),
    saturation_minimum: float = Query(0.1, ge=0.0, le=1.0, description="Min saturation"),
    intensity_upper_limit: float = Query(0.9, ge=0.0, le=1.0, description="Intensity upper limit"),
    intensity_lower_limit: float = Query(0.05, ge=0.0, le=1.0, description="Intensity lower limit")
):
    """
    Get positive pixel intensities without threshold classification.
    
    This allows frontend to reclassify in real-time when thresholds change.
    Returns intensity values and pixel positions for all positive pixels.
    """
    try:
        result = get_positive_pixel_intensities(
            item_id=item_id,
            hue_value=hue_value,
            hue_width=hue_width,
            saturation_minimum=saturation_minimum,
            intensity_upper_limit=intensity_upper_limit,
            intensity_lower_limit=intensity_lower_limit,
            thumbnail_width=thumbnail_width
        )
        return result
    except ValueError as e:
        logger.error(f"Intensity map error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Intensity map generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Intensity map generation failed: {str(e)}")


@router.post("/clear-cache")
async def clear_ppc_cache():
    """
    Clear the PPC cache (thumbnails and PPC results).
    
    Uses joblib Memory.clear() to invalidate cached results.
    Useful for debugging and when images are updated.
    """
    cache_dir = settings.CACHE_DIR
    
    try:
        # Clear joblib Memory cache - this invalidates all cached function results
        # Joblib handles the actual file cleanup internally
        memory.clear(warn=False)
        
        logger.info(f"PPC cache cleared: {cache_dir}")
        return {
            "success": True,
            "message": "PPC cache cleared successfully",
            "cache_dir": cache_dir
        }
    except Exception as e:
        logger.error(f"Error clearing PPC cache: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )
