"""Holistic estimator for combined pose, hand, and face landmarks.

This package provides the HolisticEstimator, which combines multiple
landmark detection types (pose, hands, face) into a single unified estimator.

The holistic estimator can process all body parts together or individually
based on configuration.

Types:
    HolisticPartLiteral: Type literal for valid holistic parts
        ("pose", "hand", "left_hand", "right_hand", "face")
"""

from typing import Literal
type HolisticPartLiteral = Literal["pose", "hand", "left_hand", "right_hand", "face"]
HOLISTIC_PARTS_LITERALS = ("pose", "hand", "left_hand", "right_hand", "face")
