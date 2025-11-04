"""Writers for saving annotated frames to files.

This package provides writers that save annotated frames (frames with
landmarks drawn on them) to video files or image sequences.

Available writers:
    - ``DummyAnnotatedFramesWriter``: No-op writer (skips saving)
    - ``Cv2AnnotatedFramesWriter``: Saves to video files using OpenCV

Writers inherit from ``BaseCollector`` and handle file creation, writing,
and cleanup based on the configured output format and apply mode
('skip', 'overwrite', 'postfix').
"""
