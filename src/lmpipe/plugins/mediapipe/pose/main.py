from enum import IntEnum
from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike as _MatLike
from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat

from lmpipe.estimator.holistic.main import ImageSlice

from ....estimator import estimate
from ....estimator.holistic.main import HolisticPoseEstimator
from ..core import get_model, MediaPipeEstimator
from .args import MediaPipePoseArgs

type MatLike = _MatLike
type NDArrayFloat = NDArray[np.floating]

class PoseNames(IntEnum):
    """Enum for pose landmark names based on MediaPipe Pose model."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class MediaPipePoseEstimator(MediaPipeEstimator, HolisticPoseEstimator):
    """MediaPipe Pose Landmarker estimator for LMPipe plugin.

    This plugin uses MediaPipe's Pose Landmarker to detect body pose landmarks in images.

    Args:
        pose_args (MediaPipePoseArgs): Arguments for configuring the pose landmarker.
        common_args (MediaPipeCommonArgs): Common arguments for all MediaPipe tasks.
    """

    hand_clip_scale = 6.0
    face_clip_scale = 4.0

    def __init__(self, pose_args: MediaPipePoseArgs.T = MediaPipePoseArgs.T()):

        if not pose_args.common_args:
            raise ValueError(
                f"pose_args.common_args must be specified ({self.__class__.__name__})"
            )

        super().__init__(common_args=pose_args.common_args)
        self.pose_args = pose_args

        self.model_asset_path = get_model("pose", pose_args.pose_model)

        self.landmarker_option = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=self.delegate
            ),
            running_mode=self.running_mode,
            num_poses=1,
            min_pose_detection_confidence=pose_args.min_pose_detection_confidence,
            min_pose_presence_confidence=pose_args.min_pose_presence_confidence,
            min_tracking_confidence=pose_args.min_pose_tracking_confidence,
        )

    def setup(self):

        self.landmarker = PoseLandmarker.create_from_options(self.landmarker_option)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(PoseNames), 4)

    @estimate
    def estimate(self, frame_src: MatLike, frame_idx: int) -> NDArrayFloat | None:

        image = Image(
            ImageFormat.SRGB,
            np.ascontiguousarray(frame_src)
        )

        detection_result = self.landmarker.detect(image) # pyright: ignore[reportUnknownMemberType]
        landmarks: list[list[NormalizedLandmark]] = detection_result.pose_landmarks  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        if not landmarks:
            return None

        return np.array([
            self._get_array_from_landmarks(lm)
            for lm in landmarks[0]
        ])

    def left_hand_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:

        wrist = landmarks[PoseNames.LEFT_WRIST]
        thumb = landmarks[PoseNames.LEFT_THUMB]
        index = landmarks[PoseNames.LEFT_INDEX]
        pinky = landmarks[PoseNames.LEFT_PINKY]

        center = (wrist + thumb + index + pinky) / 4

        d1 = np.abs(wrist - thumb)
        d2 = np.abs(wrist - index)
        d_max = np.max([d1, d2], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.hand_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.hand_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            left = int(
                np.clip(center[0] - self.hand_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
            right = int(
                np.clip(center[0] + self.hand_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
        except ValueError:
            return None

        return (top, bottom, left, right)

    def right_hand_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:

        wrist = landmarks[PoseNames.RIGHT_WRIST]
        thumb = landmarks[PoseNames.RIGHT_THUMB]
        index = landmarks[PoseNames.RIGHT_INDEX]
        pinky = landmarks[PoseNames.RIGHT_PINKY]

        center = (wrist + thumb + index + pinky) / 4

        d1 = np.abs(wrist - thumb)
        d2 = np.abs(wrist - index)
        d_max = np.max([d1, d2], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.hand_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.hand_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            left = int(
                np.clip(center[0] - self.hand_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
            right = int(
                np.clip(center[0] + self.hand_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
        except ValueError:
            return None

        return (top, bottom, left, right)

    def face_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:

        left_eye_outer = landmarks[PoseNames.LEFT_EYE_OUTER]
        right_eye_outer = landmarks[PoseNames.RIGHT_EYE_OUTER]
        mouth_left = landmarks[PoseNames.MOUTH_LEFT]
        mouth_right = landmarks[PoseNames.MOUTH_RIGHT]

        center = (left_eye_outer + right_eye_outer) / 2

        d_eye = np.abs(left_eye_outer - right_eye_outer)
        d_mouth = np.abs(mouth_left - mouth_right)
        d_max = np.max([d_eye, d_mouth], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.face_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.face_clip_scale * d_max[1], 0, 1)
                * frame_src.shape[0]
            )
            left = int(
                np.clip(center[0] - self.face_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
            right = int(
                np.clip(center[0] + self.face_clip_scale * d_max[0], 0, 1)
                * frame_src.shape[1]
            )
        except ValueError:
            return None

        return (top, bottom, left, right)
