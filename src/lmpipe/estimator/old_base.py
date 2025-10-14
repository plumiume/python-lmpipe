from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from typing import Callable as C
from abc import ABC, abstractmethod
from functools import cache
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

class decorator[E: "Estimator", **P, R](ABC):

    def __init__(self, func: C[Concatenate[E, P], R]):
        self._func = func
        self._bound = None

    def _with_bound(self, instance: E) -> Self:
        self._bound = instance
        return self

    def _get_instance(self, msg: str = "{module}:{qualname} must be called on an instance.") -> E:
        if self._bound is None:
            raise RuntimeError(msg.format(
                module=self._func.__module__,
                qualname=self._func.__qualname__,
            ))
        return self._bound

    def __get__(self, instance: E | None, owner: type[E]):

        if instance is None:
            return self

        return self.__class__(self._func)._with_bound(instance)

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class headers[E: "Estimator"](decorator[E, [], ArrayLikeStr]):

    def __call__(self, inst: E) -> NDArrayStr:
        return np.asarray(self._func(inst))

class estimate[E: "Estimator"](decorator[E, [MatLike, int], ArrayLikeFloat | None]):

    def __call__(self, frame_src: MatLike | None, frame_idx: int) -> NDArrayFloat:

        inst = self._get_instance()

        if frame_src is None:
            return np.full(inst.shape, np.nan)

        landmarks = self._func(inst, frame_src, frame_idx)

        if landmarks is None:
            return np.full(inst.shape, np.nan)

        return np.asarray(landmarks)

class annotate[E: "Estimator"](decorator[E, [MatLike, int, NDArrayFloat], MatLike | None]):

    def __call__(self, frame_src: MatLike, frame_idx: int, landmarks: NDArrayFloat) -> MatLike:

        inst = self._get_instance()

        annotated_frame = self._func(
            inst,
            frame_src,
            frame_idx,
            (np.asarray(landmarks))
        )

        if annotated_frame is None:
            return frame_src

        return annotated_frame

class Estimator(ABC):

    lmpipe_options: LMPipeOptions = DEFAULT_LMPIPE_OPTIONS

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]: ...

    @property
    @headers
    @cache
    def headers(self) -> NDArrayStr:
        return np.array([
            '.'.join(map(str, coord))
            for coord in np.ndindex(self.shape)
        ]).reshape(self.shape)

    @estimate
    @abstractmethod
    def estimate(*args: Any, **kwargs: Any) -> Any: ...

    @annotate
    def annotate(
        self,
        frame_src: MatLike,
        frame_idx: int,
        landmarks: NDArrayFloat
        ) -> MatLike | None:
        return frame_src

    def setup(self): pass

    def on_before_estimate(self, info: Any): pass # TODO

    def on_after_estimate(self, info: Any): pass # TODO
