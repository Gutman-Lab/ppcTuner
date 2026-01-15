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
    """Get image information including dimensions from DSA tiles endpoint"""
    try:
        dsa_client = get_dsa_client()
        if not dsa_client or not dsa_client.client:
            raise HTTPException(
                status_code=500,
                detail="DSA client not available"
            )
        
        # Get item info
        try:
            item_info = dsa_client.client.getItem(image_id)
        except Exception as e:
            logger.error(f"Failed to get item info for {image_id}: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Item not found: {image_id}"
            )
        
        # Get image dimensions from tiles endpoint
        # DSA tiles endpoint returns metadata including sizeX and sizeY
        try:
            tiles_info = dsa_client.client.get(f"item/{image_id}/tiles")
            width = tiles_info.get('sizeX') or tiles_info.get('width')
            height = tiles_info.get('sizeY') or tiles_info.get('height')
        except Exception as e:
            logger.warning(f"Could not get tiles info for {image_id}: {e}, trying DZI descriptor")
            # Fallback: try to parse DZI XML
            try:
                dzi_response = dsa_client.client.get(f"item/{image_id}/tiles/dzi.dzi", jsonResp=False)
                # DZI XML format: <Image ... Width="..." Height="...">
                import xml.etree.ElementTree as ET
                root = ET.fromstring(dzi_response.content)
                width = int(root.get('Width', 0))
                height = int(root.get('Height', 0))
            except Exception as e2:
                logger.error(f"Could not get dimensions from DZI for {image_id}: {e2}")
                width = None
                height = None
        
        return {
            "id": image_id,
            "name": item_info.get('name', 'Unknown'),
            "width": width,
            "height": height
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting image info for {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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


@router.get("/{image_id}/region")
async def get_image_region(
    image_id: str,
    x: float = Query(..., ge=0.0, le=1.0, description="Left edge of region (0-1, normalized)"),
    y: float = Query(..., ge=0.0, le=1.0, description="Top edge of region (0-1, normalized)"),
    width: float = Query(..., ge=0.0, le=1.0, description="Width of region (0-1, normalized)"),
    height: float = Query(..., ge=0.0, le=1.0, description="Height of region (0-1, normalized)"),
    output_width: int = Query(1024, ge=256, le=4096, description="Output image width in pixels")
):
    """
    Get a cropped region of the image based on normalized coordinates (0-1).
    
    Uses DSA's /item/{id}/tiles/region endpoint via girder_client.
    Coordinates are normalized to the full image dimensions (fraction units):
    - (0, 0) = top-left corner
    - (1, 1) = bottom-right corner
    
    The region is fetched directly from DSA at the appropriate resolution level
    and scaled to the requested output_width (aspect ratio preserved).
    """
    try:
        dsa_client = get_dsa_client()
        if not dsa_client or not dsa_client.client:
            raise HTTPException(
                status_code=500,
                detail="DSA client not available"
            )
        
        # Use girder_client to fetch region directly from DSA
        # DSA region endpoint: /item/{id}/tiles/region
        # See: https://girder.readthedocs.io/en/latest/api-docs.html#get-item-itemid-tiles-region
        # 
        # Parameters:
        #   - left, top: coordinates (0-based, can use fraction 0-1)
        #   - regionWidth, regionHeight: size (can use fraction 0-1)
        #   - units: 'fraction' (0-1), 'base_pixels', 'pixels', 'mm'
        #   - width, height: output dimensions in pixels (max width/height, aspect ratio preserved)
        #   - encoding: 'JPEG', 'PNG', 'TILED', 'Pickle'
        #   - jpegQuality: 0-100
        #   - exact: boolean (if magnification/mm specified, must match exactly)
        
        region_url = f"item/{image_id}/tiles/region"
        
        # Build parameters for region request
        # Use 'fraction' units (0-1) which is what OpenSeadragon provides via getBounds()
        params = {
            'left': x,
            'top': y,
            'regionWidth': width,
            'regionHeight': height,
            'units': 'fraction',  # Use fraction (0-1) for normalized coordinates
            'encoding': 'PNG',  # Use PNG for better quality (no compression artifacts)
        }
        
        # If output_width is specified, set the maximum output width
        # DSA will automatically preserve aspect ratio (may be smaller in one dimension)
        if output_width:
            params['width'] = output_width
            # Note: We don't specify height - DSA preserves aspect ratio automatically
            # If both width and height are specified, the image may be smaller in one dimension
        
        # Fetch region using girder_client
        # Use jsonResp=False to get binary image data
        try:
            region_response = dsa_client.client.get(region_url, parameters=params, jsonResp=False)
            image_data = region_response.content
            
            # Determine content type from response headers
            content_type = region_response.headers.get('Content-Type', 'image/png')
            
            return Response(
                content=image_data,
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=3600",
                }
            )
        except Exception as e:
            logger.error(f"Failed to fetch region from DSA for {image_id}: {e}")
            # Check if it's a 404 or other specific error
            if hasattr(e, 'status') and e.status == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Item or region not found: {image_id}"
                )
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch region from DSA: {str(e)}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch image for region extraction {image_id}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch image from DSA: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error extracting region for {image_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
