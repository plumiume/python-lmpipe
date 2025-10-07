from typing import Literal
from clipar import group, mixin # pyright: ignore[reportMissingTypeStubs]

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
