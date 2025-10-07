from typing import Literal, NamedTuple
from enum import IntEnum
import numpy as np
from cv2.typing import MatLike
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe import Image, ImageFormat

from ...landmarks import Landmarks
from .core import get_model, MediaPipeEstimator
from .face_args import FaceArgs

FACE_LANDMARKS_NUM = 468

class MediaPipeFaceEstimator(MediaPipeEstimator):

    def __init__(
        self,
        face_args: FaceArgs.T = FaceArgs.T(),
        ):

        if not face_args.common_args:
            raise ValueError(
                f"face_args.common_args must be specified ({self.__class__.__name__})"
            )

        super().__init__(common_args=face_args.common_args)
        self.face_args = face_args

        self.model_asset_path = get_model("face", face_args.face_model)

        self.landmarker_options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=self.delegate
            ),
            running_mode=self.running_mode,
            num_faces=1,
            min_face_detection_confidence=face_args.min_face_detection_confidence,
            min_face_presence_confidence=face_args.min_face_presence_confidence,
            min_tracking_confidence=face_args.min_face_tracking_confidence,
        )

    def setup(self):

        self.landmarker = FaceLandmarker.create_from_options(
            self.landmarker_options
        )

    def estimate(self, frame: MatLike | None, count: int) -> Landmarks:

        if frame is None:
            data = np.full((FACE_LANDMARKS_NUM, len(self.DIMENSIONS)), np.nan)
            return Landmarks("face", None, data)

        cv2_image = np.ascontiguousarray(frame)
        mp_image = Image(ImageFormat.SRGB, data=cv2_image)

        face_landmarker_result = self.landmarker.detect(mp_image)

        landmarks = face_landmarker_result.face_landmarks

        if landmarks:
            data = np.array([
                [getattr(lm, dim) for dim in self.DIMENSIONS]
                for lm in landmarks
            ])

        else:
            data = np.full((FACE_LANDMARKS_NUM, len(self.DIMENSIONS)), np.nan)

        return Landmarks("face", frame, data)

