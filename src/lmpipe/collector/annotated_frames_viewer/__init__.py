"""Viewers for displaying annotated frames in real-time.

This package provides viewers that display annotated frames during processing,
typically used for live preview or debugging.

Available viewers:
    - ``DummyAnnotatedFramesViewer``: No-op viewer (no display)
    - ``Cv2AnnotatedFramesViewer``: Displays frames using OpenCV imshow

Viewers inherit from ``BaseCollector`` and handle frame display without
saving to disk. They are useful for monitoring pipeline execution in real-time.
"""
