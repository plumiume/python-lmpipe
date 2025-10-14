"""
Evaluation configuration TypedDict definitions for MMPose.

Provides type definitions for metrics, evaluators, and evaluation-related settings.
"""

from typing import Dict, Any, Union, List, Optional, Literal, TypedDict
from .base import ConfigDict

__all__ = ['EvaluatorConfig', 'CocoMetricConfig', 'PCKAccuracyConfig', 'AUCConfig', 'EPEConfig']


class CocoMetricConfig(TypedDict, total=False):
    """COCO-style evaluation metric configuration."""
    type: Literal['CocoMetric']
    ann_file: Optional[str]  # Path to annotation file for evaluation
    use_area: Optional[bool]  # Whether to use area information in evaluation
    iou_type: Optional[str]  # IoU type for evaluation ('keypoints')
    prefix: Optional[str]  # Prefix for metric names in logs
    score_mode: Optional[Literal['bbox', 'bbox_keypoint', 'keypoint']]  # Scoring mode
    keypoint_score_thr: Optional[float]  # Keypoint confidence threshold
    nms_mode: Optional[Literal['oks_nms', 'soft_oks_nms']]  # Non-maximum suppression mode
    nms_thr: Optional[float]  # NMS threshold
    format_only: Optional[bool]  # Whether to format outputs only without evaluation


class PCKAccuracyConfig(TypedDict, total=False):
    """Percentage of Correct Keypoints (PCK) accuracy metric configuration."""
    type: Literal['PCKAccuracy']
    thr: Optional[float]  # Threshold for PCK calculation (default: 0.2)
    normalize: Optional[Literal['torso', 'bbox', 'head']]  # Normalization method for PCK
    prefix: Optional[str]  # Prefix for metric names in logs


class AUCConfig(TypedDict, total=False):
    """Area Under Curve (AUC) metric configuration."""
    type: Literal['AUC']
    num_thrs: Optional[int]  # Number of thresholds for AUC calculation
    normalize: Optional[Literal['torso', 'bbox', 'head']]  # Normalization method
    prefix: Optional[str]  # Prefix for metric names in logs


class EPEConfig(TypedDict, total=False):
    """End Point Error (EPE) metric configuration for 3D pose estimation."""
    type: Literal['EPE']
    mode: Optional[Literal['2d', '3d']]  # Evaluation mode (2D or 3D)
    prefix: Optional[str]  # Prefix for metric names in logs


class NMEConfig(TypedDict, total=False):
    """Normalized Mean Error (NME) metric configuration."""
    type: Literal['NME']
    normalize: Optional[Literal['inter_ocular', 'inter_pupil', 'bbox']]  # Normalization method
    prefix: Optional[str]  # Prefix for metric names in logs


class JhmdbPCKAccuracyConfig(TypedDict, total=False):
    """JHMDB-style PCK accuracy metric configuration."""
    type: Literal['JhmdbPCKAccuracy']
    thr: Optional[float]  # Threshold for PCK calculation
    prefix: Optional[str]  # Prefix for metric names in logs


class MpiiPCKAccuracyConfig(TypedDict, total=False):
    """MPII-style PCK accuracy metric configuration.""" 
    type: Literal['MpiiPCKAccuracy']
    thr: Optional[float]  # Threshold for PCK calculation
    normalize: Optional[Literal['head']]  # Normalization method (head size)
    prefix: Optional[str]  # Prefix for metric names in logs


EvaluatorConfig = Union[
    CocoMetricConfig, PCKAccuracyConfig, AUCConfig, EPEConfig, NMEConfig,
    JhmdbPCKAccuracyConfig, MpiiPCKAccuracyConfig
]  # Union of all available evaluator configurations


class EvaluationConfig(TypedDict, total=False):
    """Complete evaluation configuration."""
    val_evaluator: Union[EvaluatorConfig, List[EvaluatorConfig]]  # Validation evaluator(s)
    test_evaluator: Union[EvaluatorConfig, List[EvaluatorConfig]]  # Test evaluator(s)