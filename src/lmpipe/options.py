from multiprocessing import cpu_count
from typing import TypedDict, Literal
import cv2
from clipar import group, mixin

from .collector.landmarks_matrix_writer import (
    FormatLiteral as LMWFormatLiteral,
    ModeLiteral as LMWModeLiteral
)
from .collector.annotated_frames_viewer import FormatLiteral as AFVFormatLiteral
from .collector.annotated_frames_writer import (
    FormatLiteral as AFWFormatLiteral,
    ModeLiteral as AFWModeLiteral
)

class OutputOptions(TypedDict):
    landmarks_matrix_save_format: LMWFormatLiteral
    landmarks_matrix_save_mode: LMWModeLiteral
    annotated_frames_show_format: AFVFormatLiteral
    annotated_frames_save_format: AFWFormatLiteral
    annotated_frames_save_mode: AFWModeLiteral
    annotated_frames_save_width: int
    annotated_frames_save_height: int
    annotated_frames_save_fps: float
    annotated_frames_save_fourcc: int
    annotated_frames_save_ext: str

class OutputOptionsPartial(TypedDict, total=False):
    landmarks_matrix_save_format: LMWFormatLiteral
    landmarks_matrix_save_mode: LMWModeLiteral
    annotated_frames_show_format: AFVFormatLiteral
    annotated_frames_save_format: AFWFormatLiteral
    annotated_frames_save_mode: AFWModeLiteral
    annotated_frames_save_width: int
    annotated_frames_save_height: int
    annotated_frames_save_fps: float
    annotated_frames_save_fourcc: int
    annotated_frames_save_ext: str

class OutputOptionsGroup(mixin.ReprMixin):
    landmarks_matrix_save_format: LMWFormatLiteral = None
    'format to save landmarks matrix, default is None'
    landmarks_matrix_save_mode: LMWModeLiteral = 'skip'
    'mode to save landmarks matrix, default is "skip"'
    annotated_frames_show_format: AFVFormatLiteral = None
    'format to show annotated frames, default is None'
    annotated_frames_save_format: AFWFormatLiteral = None
    'format to save annotated frames, default is None'
    annotated_frames_save_mode: AFWModeLiteral = 'skip'
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
    max_workers: int
    executor_mode: Literal["sample", "batch"] | None
class ExecutorOptionsPartial(TypedDict, total=False):
    max_workers: int
    executor_mode: Literal["sample", "batch"] | None

class ExecutorOptionsGroup(mixin.ReprMixin):
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
    pass

class LMPipeOptionsPartial(
    OutputOptionsPartial,
    ExecutorOptionsPartial,
    total=False
    ):
    pass

@group
class LMPipeOptionsGroup(
    OutputOptionsGroup,
    ExecutorOptionsGroup
    ):
    pass

DEFAULT_LMPIPE_OPTIONS: LMPipeOptions = {
    **DEFAULT_OUTPUT_OPTIONS,
    **DEFAULT_EXECUTOR_OPTIONS,
}
