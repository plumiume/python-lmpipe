"""
Base TypedDict definitions for MMPose configuration.

Provides fundamental types and base classes used throughout the configuration system.
"""

from typing import Dict, Any, TypedDict

__all__ = ['BaseConfig', 'ConfigDict']


class BaseConfig(TypedDict, total=False):
    """Base configuration class for all MMPose configuration components."""
    type: str  # Component type identifier (e.g., 'TopdownPoseEstimator', 'CocoDataset')


ConfigDict = Dict[str, Any]  # Generic configuration dictionary type for flexible config values