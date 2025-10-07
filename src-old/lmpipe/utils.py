from typing import Callable, Any, Iterator, Iterable, Protocol
from pathlib import Path
import numpy as np
import cv2
from cv2.typing import MatLike
from concurrent.futures import Executor, Future

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

type PathLike = Path | str

class PlaceholderException(Exception):
    pass

def is_image_file(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTS

def is_video_file(file: Path) -> bool:
    return file.suffix.lower() in VIDEO_EXTS

def capture_to_frames(capture: cv2.VideoCapture) -> Iterator[MatLike]:
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        yield frame

def imgseq_pathes_to_frames(pathes: Iterable[Path]) -> Iterator[MatLike]:
    for f in pathes:

        try:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        except Exception:
            img = None

        if img is None:
            try:
                img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img = None

        if img is None:
            raise PlaceholderException

        yield img


class MapExecutor(Executor):

    def __init__(
        self,
        max_workers: int | None = None, # not used
        initializer: Callable[..., None] | None = None,
        initargs: tuple[Any, ...] = ()
        ):
        if initializer is not None:
            initializer(*initargs)

    def submit[**P, T](
        self,
        fn: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
        ) -> Future[T]:

        ftr = Future[T]()

        try:
            ret = fn(*args, **kwargs)
        except Exception as e:
            ftr.set_exception(e)
        else:
            ftr.set_result(ret)

        return ftr

    def map[T](
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **not_use: Any
        ) -> Iterator[T]:

        return map(fn, *iterables)
