from abc import ABC, abstractmethod
from typing import Any, Callable


from functools import cache, update_wrapper
from numpy._typing import (
    _ArrayLikeFloat_co as ArrayLikeFloat, # pyright: ignore[reportPrivateUsage]
    _ArrayLikeStr_co as ArrayLikeStr, # pyright: ignore[reportPrivateUsage]
)
from numpy.typing import NDArray
import numpy as np
from cv2.typing import MatLike

from ..options import LMPipeOptions, DEFAULT_LMPIPE_OPTIONS

type NDArrayFloat = NDArray[np.floating]
type NDArrayStr = NDArray[np.str_]
type Estimator = 'Estimator' # pyright: ignore[reportRedeclaration]

def headers[E: Estimator](
    func: Callable[[E], ArrayLikeStr]
    ):
    """
    Decorator for methods that generate header arrays.

    Wraps the `headers` method of `Estimator` class to convert `ArrayLikeStr` to `NDArrayStr`.

    Args:
        func: `(self) -> ArrayLikeStr`:
            Method that returns header array

    Returns:
        out: `(self) -> NDArrayStr`:
            Wrapped method
    """

    def wrapper(self: E) -> NDArrayStr:
        return np.asarray(func(self))

    update_wrapper(wrapper, func)
    return wrapper

def estimate[E: Estimator](
    func: Callable[[E, MatLike, int], ArrayLikeFloat | None]
    ):
    """
    Decorator for methods that perform estimation processing.

    Returns an array filled with NaN values when the frame is `None` or landmarks are `None`.
    Converts the return value to `NDArrayFloat`.

    Args:
        func: `(E, MatLike, int) -> ArrayLikeFloat | None`:
            Method that performs estimation processing

    Returns:
        out: `(E, MatLike | None, int) -> NDArrayFloat`:
            Wrapped method
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
    """
    Decorator for methods that annotate landmarks on frames.

    Returns the original frame when the annotation result is None.

    Args:
        func: `(E, MatLike, int, NDArrayFloat) -> MatLike | None`:
            Method that performs annotation processing

    Returns:
        out: `(E, MatLike, int, NDArrayFloat) -> MatLike`:
            Wrapped method
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
        """
        Abstract property that returns shape of the estimation result array.

        Returns:
            out: `tuple[int, int]`:
                Shape of estimation result array (rows, columns)
        """
        ...

    @estimate
    @abstractmethod
    def estimate(
        self,
        frame_src: MatLike,
        frame_idx: int
        ) -> ArrayLikeFloat | None:
        """
        Abstract method that estimates landmarks from frames.

        Args:
            frame_src: `MatLike`
                Input frame image
            frame_idx: `int`
                Frame index

        Returns:
            out: `ArrayLikeFloat | None`
                Estimated landmark array, or `None` if estimation failed
        """
        ...

    @property
    @headers
    @cache
    def headers(self) -> ArrayLikeStr:
        """
        Returns array of header names corresponding to each element of estimation result.

        Returns:
            out: `ArrayLikeStr`:
                Array of header names matching the shape of estimation results

        Generates coordinate indices as dot-separated strings based on `shape`.
        Results are cached.

        Current Implementation:
            Creates headers in format "row.col" (e.g., "0.0", "0.1", "1.0", "1.1")
            based on coordinate indices from np.ndindex(self.shape).
            For a shape of (2, 3), generates: ["0.0", "0.1", "0.2", "1.0", "1.1", "1.2"]
            reshaped to match the original shape.

        Override Guidelines:
            This method can be overridden to provide custom header names.
            When overriding, ensure all three decorators are applied in the correct order:

            ```python
            @property
            @headers
            @cache
            def headers(self) -> ArrayLikeStr:
                # Custom implementation
                return custom_header_array
            ```

            The @headers decorator converts ArrayLikeStr to NDArrayStr.
            The @cache decorator ensures results are cached for performance.
        """

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
        """
        Annotates landmarks on frames.

        Args:
            frame_src: `MatLike`
                Input frame image to be annotated
            frame_idx: `int`
                Frame index in the sequence (can be used for temporal annotations)
            landmarks: `NDArrayFloat`
                Landmark array containing the estimated landmark coordinates

        Returns:
            out: `MatLike | None`:
                Annotated frame with landmarks drawn, or `None` to use original frame

        Default implementation does nothing and returns `None`.
        Override in subclasses to implement specific annotation processing.

        Current Implementation:
            Returns None, indicating no annotation is performed.
            The @annotate decorator will return the original frame_src unchanged.
            This serves as a pass-through implementation for subclasses to override.

        Override Guidelines:
            This method can be overridden to provide custom annotation visualization.
            When overriding, ensure the @annotate decorator is applied:
            
            ```python
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
            ```

            The @annotate decorator handles None return values by returning the original frame.
            If your implementation returns None, the original frame will be returned unchanged.
        """
        return None

    def setup(self):
        """
        Optional setup method for initialization tasks.

        This method can be overridden in subclasses to perform any necessary setup or initialization
        before the estimator is used. It is called once after the estimator instance is created.

        Default implementation does nothing.
        """
        pass

    def on_before_estimate(self, info: Any):
        """
        Optional hook method called before each estimation.

        This method is called on each worker when executing the first task (corresponding to a function call)
        of a job (corresponding to a submit or map call).

        This method can be overridden in subclasses to perform any actions or logging
        before the `estimate` method is called. It receives an `info` object that can
        contain relevant context or metadata about the upcoming estimation.

        Args:
            info: `Any`
                Contextual information or metadata about the upcoming estimation.
        Default implementation does nothing.
        """
        pass

    def on_after_estimate(self, info: Any):
        """
        Optional hook method called after each estimation.

        This method is called on the job submitter (caller) when the job has finished.

        This method can be overridden in subclasses to perform any actions or logging
        after the `estimate` method has been called. It receives an `info` object that can
        contain relevant context or metadata about the completed estimation.

        Args:
            info: `Any`
                Contextual information or metadata about the completed estimation.
        Default implementation does nothing.
        """
        pass

    @cache
    def configure_missing_array(self):
        """Configure the missing array for landmark estimation.

        This method creates a missing array filled with the missing value for each landmark.
        The missing value is used to indicate the absence of a landmark in the estimation.

        Returns:
            NDArrayFloat: An array filled with the missing value for each landmark.
        """
        return np.full(self.shape, self.missing_value)
