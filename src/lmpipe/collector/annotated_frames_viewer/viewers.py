from abc import ABC, abstractmethod

import cv2
from cv2.typing import MatLike

from ..base import BaseCollector, ProcessFrameResult, LMPipeOptions

class AnnotatedFramesViewer(BaseCollector, ABC):

    def collect(self, result: ProcessFrameResult):
        return self.collect_frame(
            result.annotated_frame,
            result.frame_idx
        )

    @abstractmethod
    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int): ...

class DummyAnnotatedFramesViewer(AnnotatedFramesViewer):

    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int):
        pass

    def close(self):
        pass

class Cv2AnnotatedFramesViewer(AnnotatedFramesViewer):

    def __init__(self, lmpipe_options: LMPipeOptions, window_name: str | None = None):

        super().__init__(lmpipe_options)

        if window_name is None:
            self.window_name = str(id(self))
        else:
            self.window_name = window_name

    def setup(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def collect_frame(self, annotated_frame: MatLike | None, frame_idx: int):
        if annotated_frame is not None:
            cv2.imshow(self.window_name, annotated_frame)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)
