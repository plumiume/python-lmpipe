from typing import Literal
from enum import IntEnum
from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike
from mediapipe.tasks.python.core.base_options import BaseOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.category import Category # pyright: ignore[reportMissingTypeStubs]
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark # pyright: ignore[reportMissingTypeStubs]
from mediapipe import Image, ImageFormat

from ....estimator import estimate
from ....estimator.holistic.main import HolisticPartEstimator
from ..core import get_model, MediaPipeEstimator
from .args import MediaPipeHandArgs

type NDArrayFloat = NDArray[np.floating]

class HandNames(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

class MediaPipeHandEstimator(MediaPipeEstimator, HolisticPartEstimator):

    def __init__(
        self,
        hand_args: MediaPipeHandArgs.T = MediaPipeHandArgs.T(),
        category: Literal["left", "right"] | None = None
        ):

        if not hand_args.common_args:
            raise ValueError(
                f"hand_args.common_args must be specified ({self.__class__.__name__})"
            )

        if category is None:
            raise ValueError("category must be specified as 'left' or 'right'")

        self.category = category

        super().__init__(common_args=hand_args.common_args)
        self.hand_args = hand_args

        self.model_asset = get_model("hand", hand_args.hand_model)

        self.landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset,
                delegate=self.delegate
            ),
            running_mode=self.running_mode,
            num_hands=2,
            min_hand_detection_confidence=hand_args.min_hand_detection_confidence,
            min_hand_presence_confidence=hand_args.min_hand_presence_confidence,
            min_tracking_confidence=hand_args.min_hand_tracking_confidence
        )

    def setup(self):

        self.landmarker = HandLandmarker.create_from_options(self.landmarker_options)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(HandNames), 4)

    @estimate
    def estimate(self, frame_src: MatLike, idx: int) -> NDArrayFloat | None:

        image = Image(
            ImageFormat.SRGB,
            np.ascontiguousarray(frame_src)
        )

        detection_result = self.landmarker.detect(image) # pyright: ignore[reportUnknownMemberType]
        landmarks: list[list[NormalizedLandmark]] = detection_result.hand_landmarks  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        handedness: list[list[Category]] = detection_result.handedness  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        primary_landmarks = None
        secondary_landmarks = None

        for ctgrs, lms in reversed(list(zip(handedness, landmarks))):

            if not ctgrs:
                continue

            if ctgrs[0].category_name == self.category:
                primary_landmarks = lms
            else:
                secondary_landmarks = lms

        if primary_landmarks is not None:
            return np.array([
                self._get_array_from_landmarks(lm)
                for lm in primary_landmarks
            ])

        if secondary_landmarks is not None:
            return np.array([
                self._get_array_from_landmarks(lm)
                for lm in secondary_landmarks
            ])

        return None