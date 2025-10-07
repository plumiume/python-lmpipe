from typing import Literal
from clipar import namespace, mixin

from .core_args import CommonArgs

@namespace
class FaceArgs(mixin.ReprMixin):
    """Configuration arguments for MediaPipe face landmark estimation.
    
    This class defines face-specific parameters including model selection
    and confidence thresholds for face detection, presence, and tracking.
    """

    common_args = CommonArgs
    "Common configuration arguments for MediaPipe estimators."

    face_model: Literal["full"] = "full"
    "The face model variant to use."
    # num_faces: int = 1
    min_face_detection_confidence: float = 0.5
    "The minimum confidence score for the face detection to be considered successful."
    min_face_presence_confidence: float = 0.5
    "The minimum confidence score of face presence score in the face landmark detection."
    min_face_tracking_confidence: float = 0.5
    "The minimum confidence score for the face tracking to be considered successful."
