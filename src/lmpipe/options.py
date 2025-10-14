"""Configuration options for the LMPipe pipeline.

This module defines TypedDict classes and default configurations for
output options, executor options, and combined LMPipe options used
throughout the pipeline processing.

Attributes:
    DEFAULT_OUTPUT_OPTIONS (OutputOptions): Default output configuration.
    DEFAULT_EXECUTOR_OPTIONS (ExecutorOptions): Default executor configuration.
    DEFAULT_LMPIPE_OPTIONS (LMPipeOptions): Combined default configuration.
"""

from multiprocessing import cpu_count
from typing import TypedDict, Literal
import cv2
from clipar import group, mixin

from .collector.base import ModeLiteral
from .collector.landmarks_matrix_writer import FormatLiteral as LMWFormatLiteral
from .collector.annotated_frames_viewer import FormatLiteral as AFVFormatLiteral
from .collector.annotated_frames_writer import FormatLiteral as AFWFormatLiteral

class OutputOptions(TypedDict):
    """Type definition for output configuration options.
    
    Attributes:
        landmarks_matrix_save_format: Format for saving landmarks matrix.
        landmarks_matrix_save_mode: Mode for saving landmarks matrix.
        annotated_frames_show_format: Format for displaying annotated frames.
        annotated_frames_save_format: Format for saving annotated frames.
        annotated_frames_save_mode: Mode for saving annotated frames.
        annotated_frames_save_width: Width for saved annotated frames.
        annotated_frames_save_height: Height for saved annotated frames.
        annotated_frames_save_fps: FPS for saved annotated frames.
        annotated_frames_save_fourcc: FourCC codec for saved video frames.
        annotated_frames_save_ext: File extension for saved annotated frames.
    """
    landmarks_matrix_save_format: LMWFormatLiteral
    landmarks_matrix_save_mode: ModeLiteral
    annotated_frames_show_format: AFVFormatLiteral
    annotated_frames_save_format: AFWFormatLiteral
    annotated_frames_save_mode: ModeLiteral
    annotated_frames_save_width: int
    annotated_frames_save_height: int
    annotated_frames_save_fps: float
    annotated_frames_save_fourcc: int
    annotated_frames_save_ext: str

class OutputOptionsPartial(TypedDict, total=False):
    """Partial type definition for output configuration options.
    
    Same as OutputOptions but with all fields optional.
    """
    landmarks_matrix_save_format: LMWFormatLiteral
    landmarks_matrix_save_mode: ModeLiteral
    annotated_frames_show_format: AFVFormatLiteral
    annotated_frames_save_format: AFWFormatLiteral
    annotated_frames_save_mode: ModeLiteral
    annotated_frames_save_width: int
    annotated_frames_save_height: int
    annotated_frames_save_fps: float
    annotated_frames_save_fourcc: int
    annotated_frames_save_ext: str

class OutputOptionsGroup(mixin.ReprMixin):
    """CLI argument group for output configuration options."""
    landmarks_matrix_save_format: LMWFormatLiteral = None
    'format to save landmarks matrix, default is None'
    landmarks_matrix_save_mode: ModeLiteral = 'skip'
    'mode to save landmarks matrix, default is "skip"'
    annotated_frames_show_format: AFVFormatLiteral = None
    'format to show annotated frames, default is None'
    annotated_frames_save_format: AFWFormatLiteral = None
    'format to save annotated frames, default is None'
    annotated_frames_save_mode: ModeLiteral = 'skip'
    'mode to save annotated frames, default is "skip"'
    annotated_frames_save_width: int = 640
    'width to save annotated frames'
    annotated_frames_save_height: int = 480
    'height to save annotated frames'
    annotated_frames_save_fps: float = 30.0
    'fps to save annotated frames'
    annotated_frames_save_fourcc: int = cv2.VideoWriter.fourcc(*'mp4v')
    'fourcc to save annotated frames, default is "mp4v"'
    annotated_frames_save_ext: str = '.mp4'
    'file extension to save annotated frames, default is ".mp4"'

DEFAULT_OUTPUT_OPTIONS: OutputOptions = {
    'landmarks_matrix_save_format': None,
    'landmarks_matrix_save_mode': 'skip',
    'annotated_frames_show_format': None,
    'annotated_frames_save_format': None,
    'annotated_frames_save_mode': 'skip',
    'annotated_frames_save_width': 640,
    'annotated_frames_save_height': 480,
    'annotated_frames_save_fps': 30.0,
    'annotated_frames_save_fourcc': cv2.VideoWriter.fourcc(*'mp4v'),
    'annotated_frames_save_ext': '.mp4',
}

class ExecutorOptions(TypedDict):
    """Type definition for executor configuration options.
    
    Attributes:
        max_workers: Maximum number of worker processes.
        executor_mode: Processing mode, either "sample" or "batch".
    """
    max_workers: int
    executor_mode: Literal["sample", "batch"] | None

class ExecutorOptionsPartial(TypedDict, total=False):
    """Partial type definition for executor configuration options.
    
    Same as ExecutorOptions but with all fields optional.
    """
    max_workers: int
    executor_mode: Literal["sample", "batch"] | None

class ExecutorOptionsGroup(mixin.ReprMixin):
    """CLI argument group for executor configuration options."""
    max_workers: int = cpu_count()
    'maximum number of workers, default is number of CPU cores'
    executor_mode: Literal["sample", "batch"] | None = None
    'executor mode, "sample" or "batch", default is None'

DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
    'max_workers': cpu_count(),
    'executor_mode': 'sample',
}

class LMPipeOptions(
    OutputOptions,
    ExecutorOptions,
    ):
    """Complete LMPipe configuration options.
    
    Combines output and executor options into a single configuration type.
    """
    pass

class LMPipeOptionsPartial(
    OutputOptionsPartial,
    ExecutorOptionsPartial,
    total=False
    ):
    """Partial LMPipe configuration options.
    
    Same as LMPipeOptions but with all fields optional, useful for
    overriding specific configuration values.
    """
    pass

@group
class LMPipeOptionsGroup(
    OutputOptionsGroup,
    ExecutorOptionsGroup
    ):
    """CLI argument group combining output and executor options."""
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_OUTPUT_OPTIONS,
    **DEFAULT_EXECUTOR_OPTIONS,
}
