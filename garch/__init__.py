"""
GARCH modeling package for volatility analysis.
Implements various GARCH models and estimation procedures.
"""

from .estimator import GARCHEstimator, GARCHResult
from .data_prep import GarchDataPrep

__all__ = ['GARCHEstimator', 'GARCHResult', 'GarchDataPrep']
