"""LMPipe - Landmark estimation pipeline framework.

This is the root package for LMPipe, a flexible framework for running
landmark estimation pipelines on videos, image sequences, and camera streams.

The package provides:
- Estimator base classes and implementations for landmark detection
- Pipeline interface for orchestrating processing workflows
- Collectors for handling various output formats (landmarks, annotated frames)
- Plugin system for extending with custom estimators
- Utilities for input detection and frame processing

Main Components:
    - ``lmpipe.interface``: Pipeline orchestration and execution management
    - ``lmpipe.estimator``: Base classes and built-in estimators
    - ``lmpipe.collector``: Output handlers for landmarks and annotated frames
    - ``lmpipe.plugins``: Plugin loader and external estimator integration
    - ``lmpipe.options``: Configuration options for pipeline execution
"""

from pathlib import Path

PATH = Path(__file__).resolve().parent
MODULE = __name__
