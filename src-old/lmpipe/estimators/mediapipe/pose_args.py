from typing import Literal
from clipar import namespace, mixin

from .core_args import CommonArgs

@namespace
class PoseArgs(mixin.ReprMixin):
    """Configuration arguments for MediaPipe pose estimation.
    
    This class defines pose-specific parameters including model selection
    and confidence thresholds for pose detection, presence, and tracking.
    """

    common_args = CommonArgs
    "Common configuration arguments for MediaPipe estimators."

    pose_model: Literal["lite", "full", "heavy"] = "full"
    "The pose model variant to use. 'lite' for speed, 'heavy' for accuracy."
    # num_poses: int = 1
    min_pose_detection_confidence: float = 0.0
    "The minimum confidence score for the pose detection to be considered successful."
    min_pose_presence_confidence: float = 0.0
    "The minimum confidence score of pose presence score in the pose landmark detection."
    min_pose_tracking_confidence: float = 0.0
    "The minimum confidence score for the pose tracking to be considered successful."
