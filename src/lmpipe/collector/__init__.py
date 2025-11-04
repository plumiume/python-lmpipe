"""Output collectors for landmarks and annotated frames.

This package provides collectors that handle the output from landmark estimation:
- Landmarks matrix writers (save to .npy, .csv, .json)
- Annotated frames writers (save to video files)
- Annotated frames viewers (display with cv2)

All collectors inherit from ``BaseCollector`` and implement the collect/close protocol.

Subpackages:
    - ``landmarks_matrix_writer``: Writers for saving landmark matrices
    - ``annotated_frames_writer``: Writers for saving annotated frames to files
    - ``annotated_frames_viewer``: Viewers for displaying annotated frames
"""
