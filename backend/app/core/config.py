"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # DSA Configuration
    DSA_BASE_URL: str = "http://bdsa.pathology.emory.edu:8080/api/v1"
    DSAKEY: str = ""
    DSA_START_FOLDER: str = ""
    DSA_START_FOLDERTYPE: str = "collection"  # "collection" or "folder"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5174",
        "http://localhost:8080",
        "http://localhost:3000",
    ]
    
    # Cache
    CACHE_DIR: str = "/app/.npCacheDir"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
