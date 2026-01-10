"""
DSA (Digital Slide Archive) client service
Wraps girder_client for use in FastAPI
"""
import os
import logging
from typing import Optional
import girder_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class DSAClient:
    """DSA client wrapper for Girder API"""
    
    def __init__(self):
        """Initialize DSA client with authentication"""
        self.base_url = settings.DSA_BASE_URL
        self.api_key = settings.DSAKEY.strip() if settings.DSAKEY else ""
        self.client: Optional[girder_client.GirderClient] = None
        self.token: Optional[str] = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with DSA using API key and get token"""
        try:
            self.client = girder_client.GirderClient(apiUrl=self.base_url)
            
            if self.api_key:
                # Authenticate with API key - this sets up the client's authentication
                response = self.client.authenticate(apiKey=self.api_key)
                # Get the actual token ID for use in headers
                try:
                    token_info = self.client.get("token/current")
                    self.token = token_info.get("_id")
                    if not self.token:
                        logger.warning("Token info missing _id, falling back to API key")
                        self.token = self.api_key
                    else:
                        logger.info(f"Authenticated with DSA, token ID: {self.token[:10]}...")
                except Exception as e:
                    logger.warning(f"Could not get token info: {e}, falling back to API key")
                    # Fallback to API key if token endpoint fails
                    self.token = self.api_key
            else:
                logger.warning("No DSAKEY provided, DSA client will have limited functionality")
                self.token = None
        except Exception as e:
            logger.error(f"Failed to authenticate with DSA: {e}")
            self.token = None
    
    def get_token(self) -> Optional[str]:
        """Get the authentication token"""
        return self.token
