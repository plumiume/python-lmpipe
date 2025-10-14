"""
Dataset configuration TypedDict definitions for MMPose.

Provides type definitions for dataset loading, data loaders, and data processing pipelines.
"""

from typing import Dict, Any, Union, List, Optional, Literal, TypedDict, Tuple
from .base import BaseConfig, ConfigDict

__all__ = [
    'DatasetConfig', 'DataLoaderConfig', 'CocoDatasetConfig', 'PipelineConfig', 'TransformConfig'
]


class TransformConfig(BaseConfig):
    """Base transform configuration for data processing pipeline."""
    pass


class LoadImageConfig(TransformConfig):
    """Load image from file path."""
    type: Literal['LoadImage']
    color_type: Optional[str]  # Color format for image loading ('color', 'grayscale', 'unchanged')
    channel_order: Optional[str]  # Channel order ('rgb', 'bgr')
    imdecode_backend: Optional[str]  # Image decoding backend ('cv2', 'pillow')


class GetBBoxCenterScaleConfig(TransformConfig):
    """Get bounding box center and scale from annotations."""
    type: Literal['GetBBoxCenterScale']
    padding: Optional[float]  # Padding factor for bounding box


class RandomFlipConfig(TransformConfig):
    """Random horizontal flip augmentation."""
    type: Literal['RandomFlip']
    direction: Optional[Literal['horizontal', 'vertical', 'both']]  # Flip direction
    prob: Optional[float]  # Probability of applying flip


class RandomHalfBodyConfig(TransformConfig):
    """Random half body augmentation for pose estimation."""
    type: Literal['RandomHalfBody']
    min_kpts: Optional[int]  # Minimum keypoints required for half body
    prob: Optional[float]  # Probability of applying half body augmentation


class RandomBBoxTransformConfig(TransformConfig):
    """Random bounding box transformation augmentation."""
    type: Literal['RandomBBoxTransform']
    scale_factor: Optional[Tuple[float, float]]  # Scale factor range for bbox scaling
    rotate_factor: Optional[float]  # Maximum rotation angle in degrees
    shift_factor: Optional[float]  # Maximum shift factor relative to bbox size


class TopdownAffineConfig(TransformConfig):
    """Topdown affine transformation to standard input size."""
    type: Literal['TopdownAffine']
    input_size: Tuple[int, int]  # Target input size as (width, height)
    use_udp: Optional[bool]  # Whether to use Unbiased Data Processing


class GenerateTargetConfig(TransformConfig):
    """Generate target heatmaps or coordinates from keypoint annotations."""
    type: Literal['GenerateTarget']
    encoder: Union[str, ConfigDict]  # Target encoder type or configuration
    target_type: Optional[str]  # Target format ('GaussianHeatmap', 'CombinedTarget')


class PackPoseInputsConfig(TransformConfig):
    """Pack inputs and targets for pose estimation models."""
    type: Literal['PackPoseInputs']
    meta_keys: Optional[List[str]]  # Metadata keys to include in packed data


class AlbumentationConfig(TransformConfig):
    """Albumentations-based data augmentation."""
    type: Literal['Albumentation']
    transforms: List[ConfigDict]  # List of Albumentations transform configurations
    bbox_params: Optional[ConfigDict]  # Bounding box parameters for albumentations
    keymap: Optional[ConfigDict]  # Key mapping for input/output


PipelineConfig = List[Union[
    LoadImageConfig, GetBBoxCenterScaleConfig, RandomFlipConfig, RandomHalfBodyConfig,
    RandomBBoxTransformConfig, TopdownAffineConfig, GenerateTargetConfig, 
    PackPoseInputsConfig, AlbumentationConfig, TransformConfig
]]  # Data processing pipeline as list of transforms


class DatasetConfig(BaseConfig):
    """Base dataset configuration."""
    data_root: Optional[str]  # Root directory path for dataset
    data_mode: Optional[Literal['topdown', 'bottomup']]  # Data processing mode
    ann_file: Optional[str]  # Annotation file path relative to data_root
    data_prefix: Optional[ConfigDict]  # Data path prefixes (e.g., {'img': 'images/'})
    pipeline: Optional[PipelineConfig]  # Data processing pipeline
    test_mode: Optional[bool]  # Whether dataset is in test mode
    metainfo: Optional[ConfigDict]  # Dataset metadata information


class CocoDatasetConfig(DatasetConfig):
    """COCO dataset configuration for pose estimation."""
    type: Literal['CocoDataset']
    bbox_file: Optional[str]  # Path to detection results file for topdown mode
    use_gt_bbox: Optional[bool]  # Whether to use ground truth bounding boxes
    bbox_thr: Optional[float]  # Bounding box confidence threshold
    nms_thr: Optional[float]  # Non-maximum suppression threshold
    soft_nms: Optional[bool]  # Whether to use soft NMS
    oks_thr: Optional[float]  # OKS threshold for duplicate removal
    vis_thr: Optional[float]  # Visibility threshold for keypoints


class CustomDatasetConfig(DatasetConfig):
    """Custom dataset configuration."""
    type: str  # Custom dataset class name
    custom_cfg: Optional[ConfigDict]  # Custom dataset-specific configuration


class SamplerConfig(TypedDict, total=False):
    """Data sampler configuration."""
    type: str  # Sampler type ('DefaultSampler', 'InfiniteSampler', etc.)
    shuffle: Optional[bool]  # Whether to shuffle data
    round_up: Optional[bool]  # Whether to round up dataset size
    seed: Optional[int]  # Random seed for sampling


class DataLoaderConfig(TypedDict, total=False):
    """Data loader configuration."""
    batch_size: int  # Batch size for data loading
    num_workers: Optional[int]  # Number of worker processes for data loading
    persistent_workers: Optional[bool]  # Whether to keep workers persistent across epochs
    sampler: Optional[SamplerConfig]  # Data sampler configuration
    dataset: Union[CocoDatasetConfig, CustomDatasetConfig, DatasetConfig]  # Dataset configuration
    collate_fn: Optional[str]  # Collate function for batching
    pin_memory: Optional[bool]  # Whether to pin memory for faster GPU transfer
    drop_last: Optional[bool]  # Whether to drop incomplete last batch
    timeout: Optional[float]  # Timeout for data loading
    worker_init_fn: Optional[str]  # Worker initialization function