"""
Image fetching utilities for DSA (Digital Slide Archive)
Handles fetching thumbnails and regions from DSA with caching
"""
import numpy as np
import logging
import requests
import os
from io import BytesIO
from PIL import Image
from typing import Optional
from joblib import Memory
from app.services.dsa_client import DSAClient
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize joblib Memory cache for numpy arrays
cache_dir = settings.CACHE_DIR
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

# Global DSA client instance (initialized on first use)
_dsa_client: Optional[DSAClient] = None


def get_dsa_client() -> DSAClient:
    """Get or create DSA client instance"""
    global _dsa_client
    if _dsa_client is None:
        _dsa_client = DSAClient()
    return _dsa_client


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
        logger.error(f"Failed to fetch thumbnail for {item_id}: {e}")
        return None


@memory.cache
def _fetch_region_uncached(item_id: str, x: float, y: float, width: float, height: float, output_width: int, token: str) -> Optional[np.ndarray]:
    """Internal cached function to fetch region image (token is part of cache key)"""
    dsa_client = get_dsa_client()
    if not dsa_client or not dsa_client.client:
        logger.error("DSA client not available for region fetch")
        return None
    
    try:
        region_url = f"item/{item_id}/tiles/region"
        params = {
            'left': x,
            'top': y,
            'regionWidth': width,
            'regionHeight': height,
            'units': 'fraction',
            'encoding': 'PNG',
            'width': output_width,
        }
        
        region_response = dsa_client.client.get(region_url, parameters=params, jsonResp=False)
        img = Image.open(BytesIO(region_response.content))
        return np.array(img)
    except Exception as e:
        logger.error(f"Failed to fetch region for {item_id}: {e}")
        return None


def get_thumbnail_image(item_id: str, width: int = 1024, dsa_client: Optional[DSAClient] = None) -> Optional[np.ndarray]:
    """Fetch thumbnail image from DSA and return as numpy array (cached)"""
    if dsa_client is None:
        dsa_client = get_dsa_client()
    token = dsa_client.get_token() or ""
    return _fetch_thumbnail_uncached(item_id, width, token)


def get_region_image(
    item_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    output_width: int = 1024,
    dsa_client: Optional[DSAClient] = None
) -> Optional[np.ndarray]:
    """Fetch region image from DSA and return as numpy array (cached)"""
    if dsa_client is None:
        dsa_client = get_dsa_client()
    token = dsa_client.get_token() or ""
    return _fetch_region_uncached(item_id, x, y, width, height, output_width, token)
