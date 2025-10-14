from abc import ABC, abstractmethod
from functools import cache
from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike

from .. import Estimator, headers, estimate, annotate

from .args import HolisticArgs

type NDArrayFloat = NDArray[np.floating]

ImageSlice = tuple[int, int, int, int] # top, bottom, left, right

class HolisticPartEstimator(Estimator, ABC):
    ...

class HolisticPoseEstimator(HolisticPartEstimator, ABC):

    @abstractmethod
    def left_hand_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:
        ...

    @abstractmethod
    def right_hand_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:
        ...

    @abstractmethod
    def face_clipfn(
        self,
        frame_src: MatLike,
        idx: int,
        landmarks: NDArrayFloat
        ) -> ImageSlice | None:
        ...

class HolisticEstimator(Estimator):

    def __init__(
        self,
        holistic_args: HolisticArgs.T,
        pose_estimator: HolisticPoseEstimator,
        left_hand_estimator: HolisticPartEstimator | None = None,
        right_hand_estimator: HolisticPartEstimator | None = None,
        face_estimator: HolisticPartEstimator | None = None
        ) -> None:

        self.holistic_args = holistic_args

        self.pose_estimator: HolisticPoseEstimator = pose_estimator
        self.left_hand_estimator: HolisticPartEstimator | None = left_hand_estimator
        self.right_hand_estimator: HolisticPartEstimator | None = right_hand_estimator
        self.face_estimator: HolisticPartEstimator | None = face_estimator

    def setup(self):

        self.pose_estimator.setup()

        if self.left_hand_estimator is not None:
            self.left_hand_estimator.setup()

        if self.right_hand_estimator is not None:
            self.right_hand_estimator.setup()

        if self.face_estimator is not None:
            self.face_estimator.setup()

    @property
    @cache
    def shape(self) -> tuple[int, int]:

        shapes = self._get_shapes()

        points = sum([s[0] for s in shapes.values()])
        dims = max([s[1] for s in shapes.values()])

        return (points, dims)

    @cache
    def _get_shapes(self) -> dict[str, tuple[int, int]]:

        return {
            "pose": self.pose_estimator.shape,
            "left_hand": (0, 0)
            if self.left_hand_estimator is None
            else self.left_hand_estimator.shape,
            "right_hand": (0, 0)
            if self.right_hand_estimator is None
            else self.right_hand_estimator.shape,
            "face": (0, 0)
            if self.face_estimator is None
            else self.face_estimator.shape
        }

    @property
    @headers
    @cache
    def headers(self):

        headers_list = [
            np.pad(
                self.pose_estimator.headers,
                ((0, 0), (0, self.shape[1] - self.pose_estimator.shape[1])),
            )
        ]

        if self.left_hand_estimator is not None:
            headers_list.append(
                np.pad(
                    self.left_hand_estimator.headers,
                    ((0, 0), (0, self.shape[1] - self.left_hand_estimator.shape[1])),
                )
            )

        if self.right_hand_estimator is not None:
            headers_list.append(
                np.pad(
                    self.right_hand_estimator.headers,
                    ((0, 0), (0, self.shape[1] - self.right_hand_estimator.shape[1])),
                )
            )

        if self.face_estimator is not None:
            headers_list.append(
                np.pad(
                    self.face_estimator.headers,
                    ((0, 0), (0, self.shape[1] - self.face_estimator.shape[1])),
                )
            )

        return np.concatenate(headers_list, 0)

    @estimate
    def estimate(self, frame_src: MatLike, frame_idx: int) -> NDArrayFloat | None:

        targets = [
            (self.left_hand_estimator, self.pose_estimator.left_hand_clipfn),
            (self.right_hand_estimator, self.pose_estimator.right_hand_clipfn),
            (self.face_estimator, self.pose_estimator.face_clipfn)
        ]

        pose_landmarks = self.pose_estimator.estimate(frame_src, frame_idx)
        ret_landmarks = [pose_landmarks]

        for estimator, clipfn in targets:

            if estimator is None:
                continue

            image_slices = clipfn(
                frame_src=frame_src,
                idx=frame_idx,
                landmarks=pose_landmarks
            )

            if image_slices is None:
                top, bottom, left, right = 0, frame_src.shape[0], 0, frame_src.shape[1]
            else:
                top, bottom, left, right = image_slices

            if bottom - top <= 0 or right - left <= 0:
                cliped = None
            else:
                cliped = frame_src[top:bottom, left:right]

            landmarks = estimator.estimate(cliped, frame_idx)

            sloop = np.array([right - left, bottom - top, 1], dtype=np.float32)
            y_inter = np.array([left, top, 0], dtype=np.float32)
            slaling = np.array([frame_src.shape[1], frame_src.shape[0], 1], dtype=np.float32)
            new_landmarks = (landmarks * sloop + y_inter) / slaling

            ret_landmarks.append(new_landmarks)

        ret_landmarks = np.concatenate(ret_landmarks, 0)

        return ret_landmarks[:, :self.holistic_args.dimensions - 1]

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: NDArrayFloat
        ) -> MatLike | None:

        shapes = self._get_shapes()

        offsets = {
            'pose': (t := 0),
            'left_hand': (t := t + shapes['pose'][0]),
            'right_hand': (t := t + shapes['left_hand'][0]),
            'face': (t := t + shapes['right_hand'][0]),
            'holistic': t + shapes['face'][0]
        }

        frame = self.pose_estimator.annotate(
            frame_src, frame_idx,
            landmarks[offsets['pose']:offsets['left_hand']]
        )

        if self.left_hand_estimator is not None:
            self.left_hand_estimator.annotate(
                frame, frame_idx,
                landmarks[offsets['left_hand']:offsets['right_hand']]
            )

        if self.right_hand_estimator is not None:
            self.right_hand_estimator.annotate(
                frame, frame_idx,
                landmarks[offsets['right_hand']:offsets['face']]
            )

        if self.face_estimator is not None:
            self.face_estimator.annotate(
                frame, frame_idx,
                landmarks[offsets['face']:offsets['holistic']]
            )

        return frame
