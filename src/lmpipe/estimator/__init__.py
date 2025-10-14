"""Estimator module for landmark estimation and annotation.

This module provides the base classes and decorators for implementing
landmark estimators that can process video frames and generate landmarks
and annotated frames.

Classes:
    Estimator: Abstract base class for all estimators.

Decorators:
    headers: Decorator for landmark header generation methods.
    estimate: Decorator for landmark estimation methods.
    annotate: Decorator for frame annotation methods.
"""

# pyright: reportUnusedImport=false
from ._base import (
    Estimator,
    headers, estimate, annotate,
)

__all__ = [
    'Estimator',
    'headers', 'estimate', 'annotate'
]
