from abc import ABC, abstractmethod
from typing import Literal
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from .base import BaseCollector, ProcessFrameResult

FormatLiteral = Literal['cv2'] | None
Formats: tuple[FormatLiteral, ...] = ('cv2', None)

ModeLiteral = Literal['skip', 'overwrite', 'postfix']
Modes: tuple[ModeLiteral, ...] = ('skip', 'overwrite', 'postfix')

class AnnotatedFramesWriter(BaseCollector, ABC):

    def collect(self, result: ProcessFrameResult):
        return self.collect_frame(
            result.annotated_frame,
            result.frame_idx
        )

    @abstractmethod
    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int): ...

class DummyAnnotatedFramesWriter(AnnotatedFramesWriter):

    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int):
        pass

    def close(self):
        pass

class Cv2AnnotatedFramesWriter(AnnotatedFramesWriter):

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        fourcc: int,
        ext: str
        ):

        self.path = path.with_suffix(ext)

        self.writer = cv2.VideoWriter(
            filename=str(self.path),
            fourcc=fourcc,
            fps=fps, 
            frameSize=(width, height)
        )

        self.dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)

    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int):

        if annotated_frame is None:
            frame = self.dummy_frame
        else:
            frame = annotated_frame

        self.writer.write(frame)

    def close(self):
        self.writer.release()
