from typing import Any, Literal
from clipar import group, mixin

@group
class CommonArgs(mixin.ReprMixin):
    """Common configuration arguments for MediaPipe estimators.
    
    This class defines shared settings that apply to all MediaPipe-based
    landmark estimators including device selection, running mode, output
    dimensions, and visualization options.
    """

    delegate: Literal["cpu", "gpu"] = "cpu"
    "The computation delegate to use for inference."
    running_mode: Literal["image", "video", "live_stream"] = "image"
    "The running mode for MediaPipe processing."
    num_workers: int = 0
    "Number of worker threads/processes for parallel processing. Default is no concurrency."

    dimensions: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
    "The dimensions of the output."

    show_visualized: bool = False
    "Whether to show the visualized output."
    save_visualized: bool = False
    "Whether to save the visualized output."

    visualized_output_ext: Literal["mp4", "av1"] = "mp4"
    "The file format for the visualized output."
    visualized_output_fourcc: str = "mp4v"
    "The fourcc code for the visualized output."
    visualized_output_fps: float = 30.0
    "The frames per second for the visualized output."
