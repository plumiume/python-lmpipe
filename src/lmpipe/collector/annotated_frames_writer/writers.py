from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike

from ..base import BaseCollector, ProcessFrameResult, LMPipeOptions

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
        lmpipe_options: LMPipeOptions,
        path: Path,
        width: int,
        height: int,
        fps: float,
        fourcc: str,
        ext: str
        ):

        super().__init__(lmpipe_options)

        self.path = self.apply_mode(
            path.with_suffix(ext),
            self.lmpipe_options['annotated_frames_save_mode']
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc

    def setup(self):

        self.writer = cv2.VideoWriter(
            filename=str(self.path),
            fourcc=cv2.VideoWriter.fourcc(*self.fourcc),
            fps=self.fps,
            frameSize=(self.width, self.height)
        )

        self.dummy_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int):

        if annotated_frame is None:
            frame = self.dummy_frame
        else:
            frame = annotated_frame

        self.writer.write(frame)

    def close(self):
        self.writer.release()
