# Compatibility for existing imports; the shared M3 implementation lives in trees.py.

from src.models.trees import PARAMS, fit_m3

__all__ = ["PARAMS", "fit_m3"]
