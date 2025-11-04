"""Command-line interface for LMPipe.

This package provides the CLI application for running landmark estimation
pipelines from the command line. It can be executed in two ways:

1. As an installed command::

       lmpipe [OPTIONS] SRC DST

2. As a Python module::

       python -m lmpipe.app.cli [OPTIONS] SRC DST

The CLI supports multiple estimator types (pose, hand, face) with plugin-based
architecture, allowing dynamic loading of estimators from external packages.

Main Components:
    - ``__main__``: Entry point with main() function
    - ``args``: Command-line argument parsing and plugin loading
    - ``progress_bar``: Progress bar management for multiprocessing
    - ``mp_rich``: Rich renderable management across processes

Example::

    # Process a video file with pose estimation
    lmpipe input.mp4 output/ --pose mediapipe

    # Process with multiple estimators
    lmpipe input.mp4 output/ --holistic mediapipe

    # Batch processing with custom workers
    lmpipe videos/ output/ --pose mediapipe --max-workers 4

Public API:
    The main() function can be imported and called programmatically::

        from lmpipe.app.cli import main
        
        # Call with sys.argv parsing
        main()

Note:
    This package is designed as both an executable module (via __main__.py)
    and an importable package, allowing flexibility in how it's used.
"""

# Re-export main function for programmatic usage
from .__main__ import main

__all__ = ['main']
