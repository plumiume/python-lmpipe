"""Writers for saving landmark matrices to files.

This package provides writers that save landmark data (coordinates and
visibility scores) to various file formats.

Available writers:
    - ``DummyLandmarksMatrixWriter``: No-op writer (skips saving)
    - ``NpyLandmarksMatrixWriter``: Saves to NumPy .npy format
    - ``CsvLandmarksMatrixWriter``: Saves to CSV format with headers
    - ``JsonLandmarksMatrixWriter``: Saves to JSON format

Writers inherit from ``BaseCollector`` and handle matrix accumulation,
file format conversion, and writing based on the configured save format
and apply mode ('skip', 'overwrite', 'postfix').
"""
