from typing import Literal
from clipar import namespace, mixin # pyright: ignore[reportMissingTypeStubs]
from ..args import CommonArgs

@namespace
class MediaPipeFaceArgs(mixin.ReprMixin):
    """Configuration arguments for MediaPipe hand landmark estimation.
    
    This class defines hand-specific parameters including model selection
    and confidence thresholds for hand detection, presence, and tracking.
    """

    common_args = CommonArgs
    "Common configuration arguments for MediaPipe estimators."

    face_model: Literal["full"] = "full"
    "The face landmark model to use. Currently, only 'full' is supported."
    # num_hands: int = 2
    min_face_detection_confidence: float = 0.5
    "The minimum confidence score for the face detection to be considered successful."
    min_face_presence_confidence: float = 0.5
    "The minimum confidence score of face presence score in the face landmark detection."
    min_face_tracking_confidence: float = 0.5
    "The minimum confidence score for the face tracking to be considered successful."
