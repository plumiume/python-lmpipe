"""Writers for saving landmark matrices in various formats.

This module provides writers for saving landmark data to different file
formats including NumPy binary files, CSV files, and JSON files.

Types:
    FormatLiteral: Supported output formats for landmark matrices.
    ModeLiteral: File handling modes (skip, overwrite, postfix).

Attributes:
    Formats: Tuple of all supported format literals.
    Modes: Tuple of all supported mode literals.

Classes:
    LandmarksMatrixWriter: Abstract base class for landmark matrix writers.
    DummyLandmarksMatrixWriter: No-op writer for testing or disabled output.
    NpyLandmarksMatrixWriter: Writer for NumPy binary format.
    CsvLandmarksMatrixWriter: Writer for CSV format.
    JsonLandmarksMatrixWriter: Writer for JSON format.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..base import BaseCollector, ProcessFrameResult, NDArrayFloat, NDArrayStr, LMPipeOptions

class LandmarksMatrixWriter(BaseCollector, ABC):
    """Abstract base class for landmark matrix writers.
    
    This class handles the collection of landmark data and delegates
    the actual writing to subclass implementations.
    """

    def collect(self, result: ProcessFrameResult):
        """Collect a frame result and extract landmark matrix data.
        
        Args:
            result (ProcessFrameResult): Frame processing result containing landmarks.
        """
        return self.collect_matrix(result.headers, result.landmarks, result.frame_idx)

    @abstractmethod
    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):
        """Collect landmark matrix data for writing.
        
        Args:
            headers (NDArrayStr): Landmark header strings.
            landmarks (NDArrayFloat): Landmark coordinate data.
            frame_idx (int): Index of the frame.
        """
        ...

class DummyLandmarksMatrixWriter(LandmarksMatrixWriter):
    """No-op writer that discards all landmark data.
    
    Used when landmark matrix output is disabled or for testing purposes.
    """

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):
        """Discard the landmark matrix data.
        
        Args:
            headers (NDArrayStr): Landmark header strings (ignored).
            landmarks (NDArrayFloat): Landmark coordinate data (ignored).
            frame_idx (int): Index of the frame (ignored).
        """
        pass

    def close(self):
        """No-op close method."""
        pass

class NpyLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, lmpipe_options: LMPipeOptions, path: Path):

        super().__init__(lmpipe_options)

        self.path = self.apply_mode(
            path.with_suffix('.npy'),
            self.lmpipe_options['landmarks_matrix_save_mode']
        )

        self.container = list[NDArrayFloat]()

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):

        self.container.append(landmarks.flatten())

    def close(self):
        np.save(self.path, np.stack(self.container))

class CsvLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, lmpipe_options: LMPipeOptions, path: Path, header: list[str] | None = None):

        super().__init__(lmpipe_options)

        self.path = self.apply_mode(
            path.with_suffix('.csv'),
            self.lmpipe_options['landmarks_matrix_save_mode']
        )

        self.header = header

    def setup(self):

        open(self.path, mode='w').close()
        self.file = open(self.path, mode='a')
        if self.header is not None:
            self.file.write(','.join(self.header))
            self.file.write('\n')

    def collect_matrix(self, headers: NDArrayStr, landmarks: NDArrayFloat, frame_idx: int):
        np.savetxt(self.file, landmarks.reshape(1, -1), delimiter=',')

    def close(self):
        self.file.close()

class JsonLandmarksMatrixWriter(LandmarksMatrixWriter):

    def __init__(self, lmpipe_options: LMPipeOptions, path: Path, header: list[str] | None = None):

        super().__init__(lmpipe_options)

        self.path = self.apply_mode(
            path.with_suffix('.json'),
            self.lmpipe_options['landmarks_matrix_save_mode']
        )

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
