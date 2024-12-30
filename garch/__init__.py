"""
GARCH modeling package for volatility analysis.
Implements various GARCH models and estimation procedures.
"""

from .estimator import GARCHEstimator
from models import GARCHResult, ForecastWindow

__all__ = ['GARCHEstimator', 'GARCHResult', 'ForecastWindow']
