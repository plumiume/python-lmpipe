from typing import Literal
from clipar import namespace, mixin # pyright: ignore[reportMissingTypeStubs]
from ..args import CommonArgs

@namespace
class MediaPipeHandArgs(mixin.ReprMixin):
    """Configuration arguments for MediaPipe hand landmark estimation.
    
    This class defines hand-specific parameters including model selection
    and confidence thresholds for hand detection, presence, and tracking.
    """

    common_args = CommonArgs
    "Common configuration arguments for MediaPipe estimators."

    hand_model: Literal["full"] = "full"
    "The hand landmark model to use. Currently, only 'full' is supported."
    # num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    "The minimum confidence score for the hand detection to be considered successful."
    min_hand_presence_confidence: float = 0.5
    "The minimum confidence score of hand presence score in the hand landmark detection."
    min_hand_tracking_confidence: float = 0.5
    "The minimum confidence score for the hand tracking to be considered successful."