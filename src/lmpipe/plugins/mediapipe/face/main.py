from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike
from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat

from ....estimator import estimate
from ....estimator.holistic.main import HolisticPartEstimator
from ..core import get_model, MediaPipeEstimator
from .args import MediaPipeFaceArgs

type NDArrayFloat = NDArray[np.floating]

FACE_LANDMARKS_NUM = 468

class MediaPipeFaceEstimator(MediaPipeEstimator, HolisticPartEstimator):
    """MediaPipe Face Landmarker estimator for LMPipe plugin.

    This plugin uses MediaPipe's Face Landmarker to detect facial landmarks in images.

    Args:
        face_args (MediaPipeFaceArgs): Arguments for configuring the face landmarker.
        common_args (MediaPipeCommonArgs): Common arguments for all MediaPipe tasks.

    """

    def __init__(
        self,
        face_args: MediaPipeFaceArgs.T = MediaPipeFaceArgs.T()
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

    @property
    def shape(self) -> tuple[int, int]:
        return (FACE_LANDMARKS_NUM, 4)

    @estimate
    def estimate(self, frame_src: MatLike, frame_idx: int) -> NDArrayFloat | None:

        image = Image(
            ImageFormat.SRGB,
            np.ascontiguousarray(frame_src)
        )

        detection_result = self.landmarker.detect(image) # pyright: ignore[reportUnknownMemberType]
        landmarks: list[list[NormalizedLandmark]] = detection_result.face_landmarks  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        if not landmarks:
            return None

        return np.array([
            self._get_array_from_landmarks(lm)
            for lm in landmarks[0]
        ])
