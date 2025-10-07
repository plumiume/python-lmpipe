from typing import Literal
from enum import IntEnum
import textwrap
import numpy as np
from cv2.typing import MatLike
import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe import Image, ImageFormat

from ...landmarks import Landmarks
from .core import get_model, MediaPipeEstimator
from .core_args import CommonArgs
from .pose_args import PoseArgs

class PoseNames(IntEnum):
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

class MediaPipePoseEstimator(MediaPipeEstimator):

    HAND_CLIP_RATIO = 6.0
    FACE_CLIP_RATIO = 4.0

    def __init__(
        self,
        pose_args: PoseArgs.T = PoseArgs.T(),
        ):

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

        import threading
        print(f"{threading.current_thread()} setup {self.__class__.__name__}")

        self.landmarker: PoseLandmarker = PoseLandmarker.create_from_options(
            self.landmarker_option
        )

    def estimate(self, frame: MatLike | None, count: int) -> Landmarks:

        if frame is None:
            data = np.full((len(PoseNames), len(self.DIMENSIONS)), np.nan)
            return Landmarks(
                "pose", None, data,
                left_hand=self.left_hand_clipfn,
                right_hand=self.right_hand_clipfn,
                face=self.face_clipfn
            )

        cv2_image = np.ascontiguousarray(frame)
        mp_image = Image(ImageFormat.SRGB, data=cv2_image)

        pose_landmarker_result = self.landmarker.detect(mp_image)

        landmarks: list[list[NormalizedLandmark]] = pose_landmarker_result.pose_landmarks

        if landmarks:
            data = np.array([
                [getattr(lm, dim) for dim in self.DIMENSIONS]
                for lm in landmarks[0]
            ])

        else:
            data = np.full((len(PoseNames), len(self.DIMENSIONS)), np.nan)

        ret_landmarks = Landmarks(
            "pose", frame, data,
            left_hand=self.left_hand_clipfn,
            right_hand=self.right_hand_clipfn,
            face=self.face_clipfn
        )

        return ret_landmarks

    def left_hand_clipfn(self, image: MatLike, array: np.ndarray):

        wrist = array[PoseNames.LEFT_WRIST]
        thumb = array[PoseNames.LEFT_THUMB]
        index = array[PoseNames.LEFT_INDEX]
        pinky = array[PoseNames.LEFT_PINKY]

        center = (wrist + thumb + index + pinky) / 4

        d1 = np.abs(thumb - pinky)
        d2 = np.abs(index - pinky)
        d_max = np.max([d1, d2], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.HAND_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.HAND_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            left = int(
                np.clip(center[0] - self.HAND_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
            right = int(
                np.clip(center[0] + self.HAND_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
        except ValueError:
            return None

        return top, bottom, left, right

    def right_hand_clipfn(self, image: MatLike, array: np.ndarray):

        wrist = array[PoseNames.RIGHT_WRIST]
        thumb = array[PoseNames.RIGHT_THUMB]
        index = array[PoseNames.RIGHT_INDEX]
        pinky = array[PoseNames.RIGHT_PINKY]

        center = (wrist + thumb + index + pinky) / 4

        d1 = np.abs(thumb - pinky)
        d2 = np.abs(index - pinky)
        d_max = np.max([d1, d2], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.HAND_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.HAND_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            left = int(
                np.clip(center[0] - self.HAND_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
            right = int(
                np.clip(center[0] + self.HAND_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
        except ValueError:
            return None

        return top, bottom, left, right

    def face_clipfn(self, image: MatLike, array: np.ndarray):

        left_eye_outer = array[PoseNames.LEFT_EYE_OUTER]
        right_eye_outer = array[PoseNames.RIGHT_EYE_OUTER]
        mouth_left = array[PoseNames.MOUTH_LEFT]
        mouth_right = array[PoseNames.MOUTH_RIGHT]

        center = (left_eye_outer + right_eye_outer) / 2

        d_eye = np.abs(left_eye_outer - right_eye_outer)
        d_mouth = np.abs(mouth_left - mouth_right)

        d_max = np.max([d_eye, d_mouth], axis=0)

        try:
            top = int(
                np.clip(center[1] - self.FACE_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            bottom = int(
                np.clip(center[1] + self.FACE_CLIP_RATIO * d_max[1], 0, 1)
                * image.shape[0]
            )
            left = int(
                np.clip(center[0] - self.FACE_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
            right = int(
                np.clip(center[0] + self.FACE_CLIP_RATIO * d_max[0], 0, 1)
                * image.shape[1]
            )
        except ValueError:
            return None

        return top, bottom, left, right
