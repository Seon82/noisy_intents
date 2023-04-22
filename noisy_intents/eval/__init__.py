__all__ = ["compute_metrics", "autodetect_device"]

from ..training.utils import autodetect_device
from .metrics import compute_metrics
