from typing import Literal, NamedTuple
from enum import IntEnum
import numpy as np
import cv2
from cv2.typing import MatLike
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe import Image, ImageFormat

from ...landmarks import Landmarks
from .core import get_model, MediaPipeEstimator
from .core_args import CommonArgs
from .hand_args import HandArgs

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

class MediaPipeHandEstimator(MediaPipeEstimator):

    def __init__(
        self,
        hand_args: HandArgs.T = HandArgs.T(),
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

        self.model_asset_path = get_model("hand", hand_args.hand_model)

        self.landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_asset_path,
                delegate=self.delegate
            ),
            running_mode=self.running_mode,
            min_hand_detection_confidence=hand_args.min_hand_detection_confidence,
            min_hand_presence_confidence=hand_args.min_hand_presence_confidence,
            min_tracking_confidence=hand_args.min_hand_tracking_confidence
        )

    def setup(self):

        self.landmarker = HandLandmarker.create_from_options(
            self.landmarker_options
        )

    def estimate(self, frame: MatLike | None, count: int) -> Landmarks:

        if frame is None:
            data = np.full((len(HandNames), len(self.DIMENSIONS)), np.nan)
            return Landmarks(f"{self.category}_hand", None, data)

        cv2_image = np.ascontiguousarray(frame)
        mp_image = Image(image_format=ImageFormat.SRGB, data=cv2_image)

        hand_landmarker_result = self.landmarker.detect(mp_image)

        handedness: list[list[Category]] = hand_landmarker_result.handedness
        landmarks: list[list[NormalizedLandmark]] = hand_landmarker_result.hand_landmarks

        primary_data = None
        secondary_data = None

        for ctgrs, lms in zip(handedness, landmarks):

            if ctgrs[0].category_name is None:
                continue

            if ctgrs[0].category_name.lower() == self.category:
                primary_data = np.array([
                    [getattr(lm, dim) for dim in self.DIMENSIONS]
                    for lm in lms
                ])

            else:
                secondary_data = np.array([
                    [getattr(lm, dim) for dim in self.DIMENSIONS]
                    for lm in lms
                ])

        if primary_data is not None:
            data = primary_data
        elif secondary_data is not None:
            data = secondary_data
        else:
            data = np.full((len(HandNames), len(self.DIMENSIONS)), np.nan)

        return Landmarks(f"{self.category}_hand", frame, data)
