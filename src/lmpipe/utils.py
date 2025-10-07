from typing import Iterator
from pathlib import Path

from cv2 import VideoCapture, imread
from cv2.typing import MatLike

type PathLike = str | Path
type SrcDst = tuple[Path, Path]

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

def is_video_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix in VIDEO_EXTS

def is_image_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix in IMAGE_EXTS

def is_image_sequence_dir(dir_path: Path) -> bool:
    return all(
        is_image_file(file) for file in dir_path.iterdir()
    )

def video_capture_frame_generator(cap: VideoCapture) -> Iterator[MatLike]:

    if not cap.isOpened():
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def image_sequence_frame_generator(dir_path: Path) -> Iterator[MatLike]:

    for file_path in dir_path.iterdir():

        if file_path.name.startswith('.'):
            continue

        frame = imread(str(file_path))

        if frame is None:
            continue

        yield frame
