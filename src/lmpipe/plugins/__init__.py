"""Plugin system for external estimator integration.

This package provides the plugin loader that discovers and registers
estimators from external packages via entry points defined in pyproject.toml.

The loader expects entry points in the format:
    ``[project.entry-points."lmpipe.plugins"]``
    ``{type}.{name} = "module:ref"``

Example:
    pose.mediapipe = "lmpipe.plugins.mediapipe:pose_entry"

The plugin system allows extending LMPipe with custom estimators without
modifying the core package.

Modules:
    - ``loader``: Dynamic plugin loading from entry points
    - ``mediapipe``: Built-in MediaPipe-based estimators
"""
