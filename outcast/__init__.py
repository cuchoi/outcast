# outcast/__init__.py
"""
outcast – conditional outlier detection
"""
from .base import BaseConditionalDetector
from .detectors.linear_regression import ConditionalLinearRegression

__all__ = [
    "BaseConditionalDetector",
    "ConditionalLinearRegression",
]
