"""
Image-related API endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel
import requests
import logging
from app.services.dsa_client import DSAClient
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Global DSA client instance
_dsa_client: Optional[DSAClient] = None


def get_dsa_client() -> DSAClient:
    """Get or create DSA client instance"""
    global _dsa_client
    if _dsa_client is None:
        _dsa_client = DSAClient()
    return _dsa_client


class ImageInfo(BaseModel):
    """Image information model"""
    id: str
    name: str
    folder_id: Optional[str] = None


@router.get("/")
async def list_images(folder_id: Optional[str] = None):
    """List images from DSA server"""
    # TODO: Implement DSA client integration
    return {"images": [], "folder_id": folder_id}


@router.get("/{image_id}")
async def get_image_info(image_id: str):
    """Get image information"""
    # TODO: Implement DSA client integration
    return {"id": image_id, "name": "Sample Image"}


@router.get("/{image_id}/thumbnail")
async def get_thumbnail(
    image_id: str,
    width: int = Query(1024, ge=64, le=4096, description="Thumbnail width in pixels")
):
    """
    Proxy endpoint to get thumbnail for an image from DSA.
    This avoids CORS issues by fetching the image on the backend.
    """
    try:
        dsa_client = get_dsa_client()
        token = dsa_client.get_token() if dsa_client else None
        
        # Build DSA thumbnail URL
        base_url = settings.DSA_BASE_URL.rstrip('/api/v1')
        url = f"{base_url}/api/v1/item/{image_id}/tiles/thumbnail?width={width}"
        if token:
            url += f"&token={token}"
        
        # Fetch thumbnail from DSA
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine content type from response headers or default to JPEG
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        
        # Return the image with proper headers
        return Response(
            content=response.content,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch thumbnail for {image_id}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch thumbnail from DSA: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error fetching thumbnail for {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{image_id}/mask")
async def get_tissue_mask(
    image_id: str,
    width: int = Query(1024, ge=64, le=4096, description="Thumbnail width in pixels"),
    background_threshold: int = Query(240, ge=0, le=255, description="Background threshold (0-255)")
):
    """
    Generate and return a tissue mask overlay image.
    Background pixels (where all RGB channels > threshold) are highlighted in red.
    """
    try:
        from io import BytesIO
        from PIL import Image as PILImage
        import numpy as np
        
        dsa_client = get_dsa_client()
        token = dsa_client.get_token() if dsa_client else None
        
        # Build DSA thumbnail URL
        base_url = settings.DSA_BASE_URL.rstrip('/api/v1')
        url = f"{base_url}/api/v1/item/{image_id}/tiles/thumbnail?width={width}"
        if token:
            url += f"&token={token}"
        
        # Fetch thumbnail from DSA
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Load image into PIL
        img = PILImage.open(BytesIO(response.content))
        img_array = np.array(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            # Grayscale, convert to RGB
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.shape[2] == 4:
            # RGBA, convert to RGB
            img_array = img_array[:, :, :3]
        
        # Ensure uint8
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # Create mask overlay: background pixels become red, tissue pixels stay original
        mask_overlay = img_array.copy()
        
        # Create background mask (pixels where all channels > threshold)
        background_mask = (img_array[:, :, 0] > background_threshold) & \
                          (img_array[:, :, 1] > background_threshold) & \
                          (img_array[:, :, 2] > background_threshold)
        
        # Set background pixels to red with some transparency
        mask_overlay[background_mask, 0] = 255  # Red
        mask_overlay[background_mask, 1] = 0    # Green
        mask_overlay[background_mask, 2] = 0    # Blue
        
        # Convert back to PIL Image
        mask_img = PILImage.fromarray(mask_overlay)
        
        # Convert to bytes
        output = BytesIO()
        mask_img.save(output, format='PNG')
        output.seek(0)
        
        # Return the mask image
        return Response(
            content=output.getvalue(),
            media_type='image/png',
            headers={
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch thumbnail for mask generation {image_id}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch thumbnail from DSA: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error generating mask for {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
