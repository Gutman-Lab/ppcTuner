"""
Optional runtime metrics utilities for PPC computations.
"""
from typing import Dict

# Try to import psutil for CPU metrics (optional)
try:
    import psutil  # type: ignore
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


def _get_cpu_metrics() -> Dict[str, float]:
    """Get current CPU usage metrics (best-effort)."""
    if HAS_PSUTIL:
        try:
            process = psutil.Process()  # type: ignore[attr-defined]
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            return {
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_mb": 0.0}
    return {"cpu_percent": 0.0, "memory_mb": 0.0}

