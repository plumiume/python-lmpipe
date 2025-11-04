from abc import ABC, abstractmethod
from typing import Any, Callable


from functools import cache, update_wrapper
from numpy._typing import (
    _ArrayLikeFloat_co, _ArrayLikeStr_co, # pyright: ignore[reportPrivateUsage]
)
from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike as _MatLike


from ..options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS

type ArrayLikeFloat = _ArrayLikeFloat_co
type ArrayLikeStr = _ArrayLikeStr_co
type MatLike = _MatLike

type NDArrayFloat = NDArray[np.floating]
type NDArrayStr = NDArray[np.str_]
type Estimator = 'Estimator' # pyright: ignore[reportRedeclaration]

def headers[E: Estimator](
    func: Callable[[E], ArrayLikeStr]
    ):
    """Decorator for landmark header generation methods.

    Args:
        func ((E) -> ArrayLikeStr): The method to be decorated.

    Returns:
        :code:`NDArrayStr`: The headers as a NumPy array.
    """

    def wrapper(self: E) -> NDArrayStr:
        return np.asarray(func(self))

    update_wrapper(wrapper, func)
    return wrapper

def estimate[E: Estimator](
    func: Callable[[E, MatLike, int], ArrayLikeFloat | None]
    ):
    """Decorator for landmark estimation methods.

    Args:
        func ((E, MatLike | None, int) -> ArrayLikeFloat | None):
            The method to be decorated.

    Returns:
        :code:`NDArrayFloat`: The estimated landmarks as a NumPy array.
    """

    def wrapper(
        self: E,
        frame_src: MatLike | None,
        frame_idx: int
        ) -> NDArrayFloat:

        if frame_src is None:
            return self.configure_missing_array()

        landmarks = func(self, frame_src, frame_idx)

        if landmarks is None:
            return self.configure_missing_array()

        return np.asarray(landmarks)

    update_wrapper(wrapper, func)
    return wrapper

def annotate[E: Estimator](
    func: Callable[[E, MatLike, int, NDArrayFloat], MatLike | None]
    ):
    """Decorator for frame annotation methods.

    Args:
        func ((E, MatLike, int, NDArrayFloat) -> MatLike | None): The method to be decorated.

    Returns:
        :code:`MatLike`: The annotated frame, or the original frame if None is returned.
    """

    def wrapper(
        self: E,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: NDArrayFloat
        ) -> MatLike:

        frame = func(self, frame_src, frame_idx, landmarks)

        if frame is None:
            return frame_src

        return frame

    update_wrapper(wrapper, func)
    return wrapper

class Estimator(ABC):

    missing_value: float = np.nan
    "Value used to indicate missing landmarks."

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS
    "LMPipe options associated with this estimator."

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Abstract property that returns shape of the estimation result array.

        Returns:
            :code:`tuple[int, int]`: The shape of the landmark array as (rows, cols).
        """
        ...

    @estimate
    @abstractmethod
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> ArrayLikeFloat | None:
        """Estimate landmarks from a video frame.

        Args:
            frame_src (MatLike): The source video frame.
            frame_idx (int): The index of the frame in the video.

        Returns:
            :code:`ArrayLikeFloat | None`: The estimated landmarks as an array-like
            structure, or None if estimation fails.

        Note:
            Generates coordinate indices as dot-separated strings based on `shape`.
            Results are cached.

        Current Implementation:
            Creates headers in format "row.col" (e.g., :code:`"0.0", "0.1", "1.0", "1.1"`)
            based on coordinate indices from np.ndindex(self.shape).
            For a shape of :code:`(2, 3)`, generates: :code:`["0.0", "0.1", "0.2", "1.0", "1.1", "1.2"]`
            reshaped to match the original shape.

        Override Guidelines:
            This method can be overridden to provide custom header names.
            When overriding, ensure all three decorators are applied in the correct order:

            When overriding, ensure all three decorators are applied in the correct order::

                @property
                @headers
                @cache
                def headers(self) -> ArrayLikeStr:
                    # Custom implementation
                    return custom_header_array

            The `@headers` decorator converts ArrayLikeStr to NDArrayStr.
            The `@cache` decorator ensures results are cached for performance.
        """
        ...

    @property
    @headers
    @cache
    def headers(self) -> ArrayLikeStr:
        """Generate headers for the landmarks."""
        
        return np.array([
            '.'.join(map(str, coord))
            for coord in np.ndindex(self.shape)
        ]).reshape(self.shape)

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: NDArrayFloat
        ) -> MatLike | None:
        """Annotate a video frame with landmarks.

        Args:
            frame_src (MatLike): The source video frame.
            frame_idx (int): The index of the frame in the video.
            landmarks (NDArrayFloat): The estimated landmarks.

        Returns:
            :code:`MatLike | None`: The annotated frame, or None to return the
            original frame.

        Default implementation does nothing and returns `None`.
        Override in subclasses to implement specific annotation processing.

        Current Implementation:
            Returns None, indicating no annotation is performed.
            The `@annotate` decorator will return the original frame_src unchanged.
            This serves as a pass-through implementation for subclasses to override.

        Override Guidelines:
            This method can be overridden to provide custom annotation visualization.
            When overriding, ensure the `@annotate` decorator is applied:

            When overriding, ensure the `@annotate` decorator is applied::

                @annotate
                def annotate(
                    self,
                    frame_src: MatLike,
                    frame_idx: int,
                    landmarks: NDArrayFloat
                    ) -> MatLike | None:

                    # Custom annotation implementation
                    annotated_frame = frame_src.copy()

                    # Draw landmarks, connections, etc.
                    return annotated_frame

            The `@annotate` decorator handles None return values by returning the original frame.
            If your implementation returns None, the original frame will be returned unchanged.
        """
        return None

    def setup(self):
        """Optional setup method for initialization tasks.

        This method can be overridden in subclasses to perform any necessary setup or initialization
        before the estimator is used. It is called once after the estimator instance is created.

        Default implementation does nothing.
        """
        pass

    def on_before_estimate(self, info: Any):
        """Optional hook method called before each estimation.

        This method is called on the job submitter (caller) before the job starts.
        This method can be overridden in subclasses to perform any setup or logging
        before the `estimate` method is called. It receives an :code:`info` object that can
        contain relevant context or metadata about the upcoming estimation.

        Args:
            info (Any): An object containing information about the upcoming estimation.
        Default implementation does nothing.
        """
        pass

    def on_after_estimate(self, info: Any):
        """Optional hook method called after each estimation.

        This method is called on the job submitter (caller) when the job has finished.

        This method can be overridden in subclasses to perform any actions or logging
        after the `estimate` method has been called. It receives an :code:`info` object that can
        contain relevant context or metadata about the completed estimation.

        Args:
            info (Any): An object containing information about the completed estimation.

        Default implementation does nothing.
        """
        pass

    @cache
    def configure_missing_array(self):
        """Configure the missing array for landmark estimation.

        This method creates a missing array filled with the missing value for each landmark.
        The missing value is used to indicate the absence of a landmark in the estimation.

        Returns:
            :code:`NDArrayFloat`: An array filled with the missing value.
        """
        return np.full(self.shape, self.missing_value)
