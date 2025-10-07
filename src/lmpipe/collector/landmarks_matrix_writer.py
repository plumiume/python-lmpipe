from abc import ABC, abstractmethod
from typing import Literal
from pathlib import Path

import numpy as np

from .base import BaseCollector, ProcessFrameResult, NDArrayFloat, NDArrayStr

FormatLiteral = Literal['.npy', '.csv', '.json'] | None
Formats: tuple[FormatLiteral, ...] = ('.npy', '.csv', '.json', None)

class LandmarksMatrixWriter(BaseCollector, ABC):

    def collect(self, result: ProcessFrameResult):
        return self.collect_matrix(result.headers, result.landmarks, result.frame_idx)

    @abstractmethod
    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int): ...

class DummyLandmarksMatrixWriter(LandmarksMatrixWriter):

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):
        pass

    def close(self):
        pass

class NpyLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, path: Path):
        self.path = path.with_suffix('.npy')
        self.container = list[NDArrayFloat]()

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):

        self.container.append(landmarks.flatten())

    def close(self):
        np.save(self.path, np.stack(self.container))

class CsvLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, path: Path, header: list[str] | None = None):

        self.path = path.with_suffix('.csv')
        open(self.path, mode='w').close()
        self.file = open(self.path, mode='a')
        if header is not None:
            self.file.write(','.join(header))
            self.file.write('\n')

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):
        np.savetxt(self.file, landmarks.reshape(1, -1), delimiter=',')

    def close(self):
        self.file.close()

class JsonLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, path: Path, header: list[str] | None = None):
        self.path = path.with_suffix('.json')
        self.header = header 
        self.container = list[NDArrayFloat]()

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):

        self.container.append(landmarks.flatten())

    def close(self):
        import json

        obj: dict[str, object] = {
            'header': self.header,
            'matrix': np.stack(self.container).tolist()
        }

        with open(self.path, mode='w') as f:
            json.dump(obj, f, indent=2)
