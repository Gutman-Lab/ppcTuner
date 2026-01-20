"""
Small PPC utility helpers.
"""
import hashlib
import numpy as np


def _get_image_hash(img: np.ndarray) -> str:
    """Generate a short hash for an image array to use as cache key."""
    return hashlib.md5(img.tobytes()).hexdigest()[:16]

