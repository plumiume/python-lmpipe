from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from cv2.typing import MatLike

type NDArrayFloat = NDArray[np.floating]
type NDArrayStr = NDArray[np.str_]

@dataclass
class ProcessFrameResult:
    frame_idx: int
    headers: NDArrayStr
    landmarks: NDArrayFloat
    annotated_frame: MatLike | None

class BaseCollector(ABC):

    @abstractmethod
    def collect(self, result: ProcessFrameResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
