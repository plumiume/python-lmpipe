"""Base classes for data collection in the LMPipe pipeline.

This module provides the abstract base class for collectors and the
data structure for frame processing results.

Types:
    NDArrayFloat: NumPy array of floating point numbers.
    NDArrayStr: NumPy array of strings.

Classes:
    ProcessFrameResult: Data structure containing frame processing results.
    BaseCollector: Abstract base class for all collectors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

from cv2.typing import MatLike

from ..options import LMPipeOptions

from .types import NDArrayFloat, NDArrayStr, ModeLiteral

@dataclass
class ProcessFrameResult:
    """Data structure for frame processing results."""
    frame_idx: int
    "Index of the processed frame."
    headers: NDArrayStr
    "Array of header strings for the landmarks."
    landmarks: NDArrayFloat
    "Array of landmark coordinates."
    annotated_frame: MatLike | None
    "Annotated frame image, if applicable."
    thread_ident: int
    "Thread identifier for the processing frame."

class BaseCollector(ABC):
    """Abstract base class for result collectors.
    
    Collectors are responsible for handling and storing the results
    of frame processing, such as saving landmarks or displaying
    annotated frames.
    """

    skip_process: bool = False
    "Flag to indicate if processing should be skipped."
    postfix_count: int = 0
    "Counter for postfixes applied to file names."

    def __init__(self, lmpipe_options: LMPipeOptions):
        self.lmpipe_options = lmpipe_options

    def setup(self):
        """Setup the collector before processing begins.
        
        This method can be overridden by subclasses to perform any
        necessary initialization before frame processing starts.
        """
        pass

    @abstractmethod
    def collect(self, result: ProcessFrameResult) -> None:
        """Collect and process a frame result.
        
        Args:
            result (ProcessFrameResult): The result of processing a single frame.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the collector and finalize any resources.
        
        This method should be called when processing is complete to
        ensure proper cleanup and finalization of output.
        """
        raise NotImplementedError

    def apply_mode(self, path: Path, mode: ModeLiteral) -> Path:
        """Apply the specified file handling mode to the given path.

        Args:
            path (Path): The original file path.
            mode (ModeLiteral): The file handling mode to apply.

        Returns:
            Path: The modified file path.
        """

        match mode:

            case 'skip':
                self.skip_process = path.exists()

            case 'overwrite':
                pass

            case 'postfix':
                new_path = path
                while new_path.exists():
                    self.postfix_count += 1
                    new_path = path.with_stem(f"{path.stem}_{self.postfix_count}")

        return path

    def apply_postfix(self, postfix_count: int):
        """Apply a postfix to the current path based on the postfix count.

        Args:
            postfix_count (int): The postfix count to apply.
        """

        path: Path | None = getattr(self, 'path', None)

        if path is None:
            return

        if postfix_count > 0:
            setattr(self, 'path', path.with_stem(f"{path.stem}_{postfix_count}"))
