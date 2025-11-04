"""Utility functions for file type detection and media processing.

This module provides utility functions for detecting file types, generating
frame iterators from video captures and image sequences, and common type
definitions used throughout the lmpipe package.

"""

from typing import Iterator
from pathlib import Path

from cv2 import VideoCapture, imread as _imread
from cv2.typing import MatLike as _MatLike

type PathLike = str | Path
type SrcDst = tuple[Path, Path]
type MatLike = _MatLike

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
"Supported video file extensions."
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
"Supported image file extensions."

def is_video_file(file_path: Path) -> bool:
    """Check if a file is a supported video file.
    
    Args:
        file_path (Path): Path to the file to check.
        
    Returns:
        :code:`bool`: True if the file is a video file, False otherwise.
    """
    return file_path.is_file() and file_path.suffix in VIDEO_EXTS

def is_image_file(file_path: Path) -> bool:
    """Check if a file is a supported image file.
    
    Args:
        file_path (Path): Path to the file to check.
        
    Returns:
        :code:`bool`: True if the file is an image file, False otherwise.
    """
    return file_path.is_file() and file_path.suffix in IMAGE_EXTS

def is_image_sequence_dir(dir_path: Path) -> bool:
    """Check if a directory contains only image files.
    
    Args:
        dir_path (Path): Path to the directory to check.
        
    Returns:
        :code:`bool`: True if the directory contains only image files, False otherwise.
    """
    return all(
        is_image_file(file) for file in dir_path.iterdir()
    )

def video_capture_frame_generator(cap: VideoCapture) -> Iterator[MatLike]:
    """Generate frames from a video capture object.
    
    Args:
        cap (VideoCapture): OpenCV VideoCapture object.
        
    Yields:
        :code:`MatLike`: Individual frames from the video.
        
    Note:
        The generator will stop when the video ends or when the capture
        is not opened.
    """

    if not cap.isOpened():
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def image_sequence_frame_generator(dir_path: Path) -> Iterator[MatLike]:
    """Generate frames from an image sequence directory.
    
    Args:
        dir_path (Path): Path to directory containing image files.
        
    Yields:
        :code:`MatLike`: Individual image frames loaded as OpenCV matrices.
        
    Note:
        Hidden files (starting with '.') are skipped. If an image cannot
        be loaded, it is skipped silently.
    """

    for file_path in dir_path.iterdir():

        if file_path.name.startswith('.'):
            continue

        frame = _imread(str(file_path))

        if frame is None:
            continue

        yield frame
