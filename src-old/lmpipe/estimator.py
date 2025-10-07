from typing import Literal, TypedDict, Any
from cv2.typing import MatLike
from pathlib import Path
from multiprocessing import cpu_count
import numpy as np
from .dst import LMDst
from .landmarks import Landmarks

class EstimatorOptions(TypedDict):
    process_type: Literal[
        "single_cpu", "multi_cpu", "stream_cpu",
        "single_gpu", "multi_gpu", "stream_gpu"
    ]
    "Type of processing to use for estimation."
    num_workers: int
    "Number of worker processes or threads to use."
    video_ext: str | None
    "File extension for output video files (e.g., 'mp4', 'avi'). None if not applicable."
    fourcc: int | None
    "FourCC code for video encoding. None if not specified."
    fps: float | None
    "Frames per second for video output. None if not specified."

class EstimatorOptionsPartial(TypedDict, total=False):
    process_type: Literal[
        "single_cpu", "multi_cpu", "stream_cpu",
        "single_gpu", "multi_gpu", "stream_gpu"
    ]
    "Type of processing to use for estimation."
    num_workers: int
    "Number of worker processes or threads to use."
    video_ext: str | None
    "File extension for output video files (e.g., 'mp4', 'avi'). None if not applicable."
    fourcc: int | None
    "FourCC code for video encoding. None if not specified."
    fps: float | None
    "Frames per second for video output. None if not specified."

ESTIMATOR_DEFAULT_OPTIONS: EstimatorOptions = {
    "process_type": "single_cpu",
    "num_workers": cpu_count() - 2,
    "video_ext": None,
    "fourcc": None,
    "fps": None
}

class Estimator:

    estimator_options: EstimatorOptions = ESTIMATOR_DEFAULT_OPTIONS
    _setuped: bool = False

    def setup(self):
        pass

    ## Main methods to implement

    def estimate(
        self,
        frame: MatLike | None,
        count: int
        ) -> Landmarks:
        raise NotImplementedError

    ## Hooks

    def on_process_sample(self, src: Path, lm_dst: LMDst):
        pass

class HolisticEstimator(Estimator):

    def __init__(
        self,
        pose_estimator: Estimator,
        left_hand_estimator: Estimator | None = None,
        right_hand_estimator: Estimator | None = None,
        face_estimator: Estimator | None = None
        ) -> None:

        self.pose_estimator = pose_estimator
        self.left_hand_estimator = left_hand_estimator
        self.right_hand_estimator = right_hand_estimator
        self.face_estimator = face_estimator

    def estimate(self, frame: MatLike | None, count: int) -> Landmarks:

        if frame is None:
            raise ValueError("Frame is None") # Root Estimator

        ret_landmarks = Landmarks()

        pose_landmarks = self.pose_estimator.estimate(frame, count)
        ret_landmarks.add_landmarks(pose_landmarks)

        targets = [
            (self.left_hand_estimator, "left_hand"),
            (self.right_hand_estimator, "right_hand"),
            (self.face_estimator, "face")
        ]

        for estimator, target in targets:

            if estimator is None:
                continue

            domain = pose_landmarks.apply_clipfn("pose", target)
            if domain is None:
                top, bottom, left, right = 0, frame.shape[0], 0, frame.shape[1]
            else:
                top, bottom, left, right = domain

            if bottom - top <= 0 or right - left <= 0:
                cliped = None
            else:
                cliped = frame[top:bottom, left:right]

            landmarks = estimator.estimate(cliped, count)

            # TODO: not use private object
            assert landmarks.item is not None
            assert landmarks._name is not None
            image, array, clip_fns = landmarks.item
            sloop= np.array([right - left, bottom - top, 1], dtype=np.float32)
            y_inter = np.array([left, top, 0], dtype=np.float32)
            slaling = np.array([frame.shape[1], frame.shape[0], 1], dtype=np.float32)
            new_array = (array * sloop + y_inter) / slaling
            new_landmarks = Landmarks(landmarks._name, image, new_array, **clip_fns)

            ret_landmarks.add_landmarks(new_landmarks)

        return ret_landmarks
