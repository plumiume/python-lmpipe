from typing import Literal
from clipar import group, mixin # pyright: ignore[reportMissingTypeStubs]

@group
class HolisticArgs(mixin.ReprMixin):
    """Configuration arguments for MediaPipe Holistic estimator.
    
    This class defines settings specific to the MediaPipe Holistic landmark
    estimator, including model selection and common arguments shared across
    MediaPipe estimators.
    """

    extra_dims: list[Literal['z', 'c']] = ['z', 'c']
    "Extra dimensions to include in the output landmarks."
