"""Configuration options for the LMPipe pipeline.

This module defines TypedDict classes and default configurations for
requiresresources, output, executor, lmpipe options used throughout the pipeline processing.

Attributes:
    DEFAULT_REQUIRES_RESOURCES (RequiresResources): Default resource requirement configuration..
    DEFAULT_OUTPUT_OPTIONS (OutputOptions): Default output configuration options..
    DEFAULT_EXECUTOR_OPTIONS (ExecutorOptions): Default executor configuration options..
    DEFAULT_LMPIPE_OPTIONS (LMPipeOptions): Combined default lmpipe configuration.
"""

from typing import TypedDict, Literal
from clipar import group, mixin
from .typedefs import ExecutorMode, ExecutorType
from .collector.base import ModeLiteral
from .collector.landmarks_matrix_writer.options import FormatLiteral as LMWFormatLiteral
from .collector.annotated_frames_viewer.options import FormatLiteral as AFVFormatLiteral
from .collector.annotated_frames_writer.options import FormatLiteral as AFWFormatLiteral

class RequiresResources(TypedDict):
    """Type definition for resource requirement configuration."""
    cpu: float
    'amount of CPU required'
    cpu_tags: list[str]
    'tags for CPU requirements'
    gpu: float
    'amount of GPU required'
    gpu_tags: list[str]
    'tags for GPU requirements'

class RequiresResourcesPartial(TypedDict, total=False):
    """Partial type definition for resource requirement configuration.
    
    Same as RequiresResources but with all fields optional.
    """
    cpu: float
    'amount of CPU required'
    cpu_tags: list[str]
    'tags for CPU requirements'
    gpu: float
    'amount of GPU required'
    gpu_tags: list[str]
    'tags for GPU requirements'

@group
class RequiresResourcesGroup(mixin.ReprMixin):
    """Type definition for resource requirement configuration."""
    cpu: float = 1.0
    'amount of CPU required'
    cpu_tags: list[str] = []
    'tags for CPU requirements'
    gpu: float = 0.0
    'amount of GPU required'
    gpu_tags: list[str] = []
    'tags for GPU requirements'

DEFAULT_REQUIRES_RESOURCES: RequiresResources = {
    'cpu': 1.0,
    'cpu_tags': [],
    'gpu': 0.0,
    'gpu_tags': []
}

class OutputOptions(TypedDict):
    """Type definition for output configuration options."""
    landmarks_matrix_save_format: LMWFormatLiteral
    'format to save landmarks matrix, default is None'
    landmarks_matrix_save_mode: ModeLiteral
    'mode to save landmarks matrix, default is "skip"'
    annotated_frames_show_format: AFVFormatLiteral
    'format to show annotated frames, default is None'
    annotated_frames_save_format: AFWFormatLiteral
    'format to save annotated frames, default is None'
    annotated_frames_save_mode: ModeLiteral
    'mode to save annotated frames, default is "skip"'
    annotated_frames_save_width: int
    'width to save annotated frames'
    annotated_frames_save_height: int
    'height to save annotated frames'
    annotated_frames_save_fps: float
    'fps to save annotated frames'
    annotated_frames_save_fourcc: str
    'fourcc to save annotated frames, default is "mp4v"'
    annotated_frames_save_ext: str
    'file extension to save annotated frames, default is ".mp4"'

class OutputOptionsPartial(TypedDict, total=False):
    """Partial type definition for output configuration options.
    
    Same as OutputOptions but with all fields optional.
    """
    landmarks_matrix_save_format: LMWFormatLiteral
    'format to save landmarks matrix, default is None'
    landmarks_matrix_save_mode: ModeLiteral
    'mode to save landmarks matrix, default is "skip"'
    annotated_frames_show_format: AFVFormatLiteral
    'format to show annotated frames, default is None'
    annotated_frames_save_format: AFWFormatLiteral
    'format to save annotated frames, default is None'
    annotated_frames_save_mode: ModeLiteral
    'mode to save annotated frames, default is "skip"'
    annotated_frames_save_width: int
    'width to save annotated frames'
    annotated_frames_save_height: int
    'height to save annotated frames'
    annotated_frames_save_fps: float
    'fps to save annotated frames'
    annotated_frames_save_fourcc: str
    'fourcc to save annotated frames, default is "mp4v"'
    annotated_frames_save_ext: str
    'file extension to save annotated frames, default is ".mp4"'

@group
class OutputOptionsGroup(mixin.ReprMixin):
    """Type definition for output configuration options."""
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
    annotated_frames_save_fourcc: str = 'mp4v'
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
    'annotated_frames_save_fourcc': 'mp4v',
    'annotated_frames_save_ext': '.mp4'
}

class ExecutorOptions(TypedDict):
    """Type definition for executor configuration options."""
    max_workers: int
    'maximum number of worker threads/processes, default is 4'
    executor_mode: ExecutorMode
    'executor mode'
    executor_type: ExecutorType
    'executor type, either "thread" or "process"'

class ExecutorOptionsPartial(TypedDict, total=False):
    """Partial type definition for executor configuration options.
    
    Same as ExecutorOptions but with all fields optional.
    """
    max_workers: int
    'maximum number of worker threads/processes, default is 4'
    executor_mode: ExecutorMode
    'executor mode'
    executor_type: ExecutorType
    'executor type, either "thread" or "process"'

@group
class ExecutorOptionsGroup(mixin.ReprMixin):
    """Type definition for executor configuration options."""
    max_workers: int = 0
    'maximum number of worker threads/processes, default is 4'
    executor_mode: ExecutorMode = None
    'executor mode'
    executor_type: ExecutorType = 'thread'
    'executor type, either "thread" or "process"'

DEFAULT_EXECUTOR_OPTIONS: ExecutorOptions = {
    'max_workers': 0,
    'executor_mode': None,
    'executor_type': 'thread'
}

class LMPipeOptions(
    OutputOptions,
    ExecutorOptions,
    RequiresResources,
    ):
    """Complete lmpipe configuration options.

Combines output and executor options into a single configuration type."""
    pass

class LMPipeOptionsPartial(
    OutputOptionsPartial,
    ExecutorOptionsPartial,
    RequiresResourcesPartial,
    total=False
    ):
    """Partial complete lmpipe configuration options.

combines output and executor options into a single configuration type.
    
    Same as LMPipeOptions but with all fields optional, useful for
    overriding specific configuration values.
    """
    pass

@group
class LMPipeOptionsGroup(
    OutputOptionsGroup.T,
    ExecutorOptionsGroup.T,
    RequiresResourcesGroup.T,
    ):
    """CLI argument group combining output and executor and requiresresources options."""
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_OUTPUT_OPTIONS,
    **DEFAULT_EXECUTOR_OPTIONS,
    **DEFAULT_REQUIRES_RESOURCES,
}
