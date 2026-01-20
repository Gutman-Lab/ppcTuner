"""
Shared joblib cache for PPC-related computations.
"""
import os
from joblib import Memory
from app.core.config import settings

# Initialize joblib Memory cache for numpy-heavy computations.
# Cache directory is persistent within Docker container.
cache_dir = settings.CACHE_DIR
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)

