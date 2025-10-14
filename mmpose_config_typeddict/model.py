"""
Model configuration TypedDict definitions for MMPose.

Provides type definitions for model components including backbones, heads, 
data preprocessors, and complete model configurations.
"""

from typing import Dict, Any, Union, List, Optional, Literal, TypedDict, Tuple
from .base import BaseConfig, ConfigDict

__all__ = [
    'ModelConfig', 'BackboneConfig', 'HeadConfig', 'DataPreprocessorConfig', 
    'TestConfig', 'RTMPoseConfig', 'ViTPoseConfig', 'RTMCCHeadConfig', 
    'TopdownHeatmapSimpleHeadConfig'
]


class DataPreprocessorConfig(BaseConfig):
    """Data preprocessor configuration for model input normalization and formatting."""
    mean: Optional[List[float]]  # RGB mean values for normalization (e.g., [123.675, 116.28, 103.53])
    std: Optional[List[float]]  # RGB standard deviation values for normalization (e.g., [58.395, 57.12, 57.375])
    bgr_to_rgb: Optional[bool]  # Whether to convert BGR input to RGB format


class TestConfig(TypedDict, total=False):
    """Test-time configuration for model inference behavior."""
    flip_test: Optional[bool]  # Whether to enable horizontal flip test time augmentation for better accuracy
    flip_mode: Optional[Literal['heatmap', 'regression']]  # Mode for handling flipped predictions
    shift_heatmap: Optional[bool]  # Whether to shift heatmap for sub-pixel accuracy improvement
    normalize: Optional[bool]  # Whether to normalize output coordinates
    post_process: Optional[str]  # Post-processing method for predictions
    modulate_kernel: Optional[int]  # Kernel size for heatmap modulation


class BackboneConfig(BaseConfig):
    """Base backbone network configuration for feature extraction."""
    init_cfg: Optional[ConfigDict]  # Weight initialization configuration
    norm_cfg: Optional[ConfigDict]  # Normalization layer configuration
    act_cfg: Optional[ConfigDict]  # Activation function configuration


class CSPNeXtConfig(BackboneConfig):
    """CSPNeXt backbone configuration for RTMPose models."""
    arch: Optional[Literal['P5', 'P6']]  # Architecture variant (P5 for smaller, P6 for larger models)
    expand_ratio: Optional[float]  # Channel expansion ratio in bottleneck blocks
    deepen_factor: Optional[float]  # Network depth scaling factor
    widen_factor: Optional[float]  # Network width scaling factor
    out_indices: Optional[Tuple[int, ...]]  # Output feature map indices to use
    channel_attention: Optional[bool]  # Whether to use channel attention mechanism


class ViTConfig(BackboneConfig):
    """Vision Transformer backbone configuration."""
    img_size: NotRequired[Tuple[int, int]]  # Input image size as (height, width)
    patch_size: NotRequired[int]  # Size of each image patch for tokenization
    embed_dim: NotRequired[int]  # Embedding dimension for transformer
    depth: NotRequired[int]  # Number of transformer layers
    num_heads: NotRequired[int]  # Number of attention heads in multi-head attention
    mlp_ratio: NotRequired[float]  # MLP hidden dimension ratio relative to embed_dim
    qkv_bias: NotRequired[bool]  # Whether to add bias to query, key, value projections
    drop_path_rate: NotRequired[float]  # Stochastic depth drop path rate
    use_checkpoint: NotRequired[bool]  # Whether to use gradient checkpointing for memory efficiency
    with_cls_token: NotRequired[bool]  # Whether to use classification token
    out_type: NotRequired[Literal['cls_token', 'featmap', 'avg_featmap']]  # Output type selection
    patch_cfg: NotRequired[ConfigDict]  # Patch embedding configuration


class ResNetConfig(BackboneConfig):
    """ResNet backbone configuration."""
    depth: NotRequired[Literal[18, 34, 50, 101, 152]]  # ResNet depth variant
    num_stages: NotRequired[int]  # Number of ResNet stages to use
    strides: NotRequired[Tuple[int, ...]]  # Stride for each stage
    dilations: NotRequired[Tuple[int, ...]]  # Dilation for each stage
    out_indices: NotRequired[Tuple[int, ...]]  # Output indices for feature maps
    style: NotRequired[Literal['pytorch', 'caffe']]  # ResNet implementation style
    deep_stem: NotRequired[bool]  # Whether to use deep stem (7x7 -> 3x3,3x3,3x3)
    avg_down: NotRequired[bool]  # Whether to use average pooling for downsampling


class HeadConfig(BaseConfig):
    """Base head configuration for pose estimation output."""
    in_channels: NotRequired[int]  # Input channel dimension from backbone
    out_channels: NotRequired[int]  # Output channel dimension (number of keypoints)
    loss: NotRequired[ConfigDict]  # Loss function configuration
    decoder: NotRequired[ConfigDict]  # Coordinate decoder configuration


class RTMCCHeadConfig(HeadConfig):
    """RTMPose coordinate classification head configuration."""
    input_size: NotRequired[Tuple[int, int]]  # Input feature map size as (height, width)
    in_featuremap_size: NotRequired[Tuple[int, int]]  # Feature map size after backbone
    simcc_split_ratio: NotRequired[float]  # Split ratio for SimCC coordinate classification
    final_layer_kernel_size: NotRequired[int]  # Kernel size for final convolution layer
    gau_cfg: NotRequired[ConfigDict]  # Gated Attention Unit configuration
    loss: NotRequired[ConfigDict]  # Loss configuration (typically KLDiscretLoss)


class TopdownHeatmapSimpleHeadConfig(HeadConfig):
    """Simple heatmap-based head configuration for topdown pose estimation."""
    num_deconv_layers: NotRequired[int]  # Number of deconvolution layers for upsampling
    num_deconv_filters: NotRequired[List[int]]  # Filter numbers for each deconv layer
    num_deconv_kernels: NotRequired[List[int]]  # Kernel sizes for each deconv layer
    extra: NotRequired[ConfigDict]  # Extra configuration (e.g., final_conv_kernel)
    upsample: NotRequired[int]  # Upsampling factor for output resolution


class ModelConfig(BaseConfig):
    """Complete model configuration for MMPose pose estimators."""
    data_preprocessor: NotRequired[DataPreprocessorConfig]  # Input data preprocessing configuration
    backbone: NotRequired[Union[CSPNeXtConfig, ViTConfig, ResNetConfig, BackboneConfig]]  # Feature extraction backbone
    head: NotRequired[Union[RTMCCHeadConfig, TopdownHeatmapSimpleHeadConfig, HeadConfig]]  # Pose estimation head
    keypoint_head: NotRequired[Union[RTMCCHeadConfig, TopdownHeatmapSimpleHeadConfig, HeadConfig]]  # Alternative head name
    neck: NotRequired[ConfigDict]  # Optional neck module between backbone and head
    test_cfg: NotRequired[TestConfig]  # Test-time inference configuration
    train_cfg: NotRequired[ConfigDict]  # Training-time configuration
    init_cfg: NotRequired[ConfigDict]  # Model weight initialization configuration


class RTMPoseConfig(ModelConfig):
    """RTMPose model configuration with CSPNeXt backbone and RTMCCHead."""
    type: Required[Literal['TopdownPoseEstimator']]  # Model type for RTMPose
    backbone: NotRequired[CSPNeXtConfig]  # CSPNeXt backbone configuration
    head: NotRequired[RTMCCHeadConfig]  # RTMCC head configuration


class ViTPoseConfig(ModelConfig):
    """ViTPose model configuration with Vision Transformer backbone."""
    type: Required[Literal['TopdownPoseEstimator']]  # Model type for ViTPose
    backbone: NotRequired[ViTConfig]  # Vision Transformer backbone configuration
    keypoint_head: NotRequired[TopdownHeatmapSimpleHeadConfig]  # Simple heatmap head configuration