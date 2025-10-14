"""
Runtime configuration TypedDict definitions for MMPose.

Provides type definitions for runtime settings, hooks, logging, and environment configuration.
"""

from typing import Dict, Any, Union, List, Optional, Literal, TypedDict
from .base import ConfigDict

__all__ = [
    'RuntimeConfig', 'DefaultHooksConfig', 'HookConfig', 'VisualizerConfig', 
    'LogProcessorConfig', 'EnvConfig'
]


class HookConfig(TypedDict, total=False):
    """Base hook configuration."""
    type: str  # Hook type name
    priority: Optional[int]  # Hook priority (lower numbers = higher priority)


class TimerHookConfig(HookConfig):
    """Timer hook for measuring training/validation time."""
    type: Literal['IterTimerHook']


class LoggerHookConfig(HookConfig):
    """Logger hook configuration for training logs."""
    type: Literal['LoggerHook']
    interval: Optional[int]  # Logging interval in iterations
    ignore_last: Optional[bool]  # Whether to ignore last incomplete interval
    reset_flag: Optional[bool]  # Whether to reset timer after logging
    by_epoch: Optional[bool]  # Whether to log by epoch or iteration


class ParamSchedulerHookConfig(HookConfig):
    """Parameter scheduler hook for learning rate scheduling."""
    type: Literal['ParamSchedulerHook']


class CheckpointHookConfig(HookConfig):
    """Checkpoint hook configuration for model saving."""
    type: Literal['CheckpointHook']
    interval: Optional[int]  # Checkpoint saving interval
    by_epoch: Optional[bool]  # Whether interval is in epochs or iterations
    save_best: Optional[str]  # Metric name for best model saving (e.g., 'coco/AP')
    rule: Optional[Literal['greater', 'less']]  # Rule for best model comparison
    max_keep_ckpts: Optional[int]  # Maximum number of checkpoints to keep
    save_last: Optional[bool]  # Whether to save last checkpoint
    save_optimizer: Optional[bool]  # Whether to save optimizer state
    out_dir: Optional[str]  # Output directory for checkpoints


class DistSamplerSeedHookConfig(HookConfig):
    """Distributed sampler seed hook for reproducible distributed training."""
    type: Literal['DistSamplerSeedHook']


class PoseVisualizationHookConfig(HookConfig):
    """Pose visualization hook for training visualization."""
    type: Literal['PoseVisualizationHook']
    enable: Optional[bool]  # Whether to enable visualization
    interval: Optional[int]  # Visualization interval in iterations
    kpt_thr: Optional[float]  # Keypoint confidence threshold for visualization
    show: Optional[bool]  # Whether to show visualizations in GUI
    wait_time: Optional[float]  # Wait time for visualization display
    out_dir: Optional[str]  # Output directory for saving visualizations


class EMAHookConfig(HookConfig):
    """Exponential Moving Average hook configuration."""
    type: Literal['EMAHook']
    ema_type: Optional[str]  # EMA type ('ExponentialMovingAverage')
    momentum: Optional[float]  # EMA momentum coefficient
    update_buffers: Optional[bool]  # Whether to update buffer parameters
    strict_load: Optional[bool]  # Whether to use strict loading for EMA weights


class DefaultHooksConfig(TypedDict, total=False):
    """Default hooks configuration."""
    timer: Optional[TimerHookConfig]  # Timer hook for time measurement
    logger: Optional[LoggerHookConfig]  # Logger hook for training logs
    param_scheduler: Optional[ParamSchedulerHookConfig]  # Parameter scheduler hook
    checkpoint: Optional[CheckpointHookConfig]  # Checkpoint saving hook
    sampler_seed: Optional[DistSamplerSeedHookConfig]  # Distributed sampler seed hook
    visualization: Optional[PoseVisualizationHookConfig]  # Visualization hook


class VisBackendConfig(TypedDict, total=False):
    """Visualization backend configuration."""
    type: str  # Backend type ('LocalVisBackend', 'TensorboardVisBackend', 'WandbVisBackend')
    save_dir: Optional[str]  # Directory for saving visualization outputs


class LocalVisBackendConfig(VisBackendConfig):
    """Local visualization backend configuration."""
    type: Literal['LocalVisBackend']


class TensorboardVisBackendConfig(VisBackendConfig):
    """TensorBoard visualization backend configuration."""
    type: Literal['TensorboardVisBackend']
    save_dir: Optional[str]  # Directory for TensorBoard logs


class WandbVisBackendConfig(VisBackendConfig):
    """Weights & Biases visualization backend configuration."""
    type: Literal['WandbVisBackend']
    init_kwargs: Optional[ConfigDict]  # W&B initialization arguments
    save_dir: Optional[str]  # Directory for W&B logs


class VisualizerConfig(TypedDict, total=False):
    """Visualizer configuration."""
    type: str  # Visualizer type ('PoseLocalVisualizer')
    vis_backends: Optional[List[Union[LocalVisBackendConfig, TensorboardVisBackendConfig, WandbVisBackendConfig]]]  # Visualization backends
    name: Optional[str]  # Visualizer name
    kpt_color: Optional[Union[str, List[int]]]  # Keypoint color specification
    link_color: Optional[Union[str, List[int]]]  # Link color specification  
    line_width: Optional[int]  # Line width for skeleton drawing
    radius: Optional[int]  # Keypoint radius for drawing
    alpha: Optional[float]  # Transparency alpha value
    show_keypoint_weight: Optional[bool]  # Whether to show keypoint confidence as size


class LogProcessorConfig(TypedDict, total=False):
    """Log processor configuration for metric averaging."""
    type: Optional[str]  # Log processor type ('LogProcessor')
    window_size: Optional[int]  # Window size for metric smoothing
    by_epoch: Optional[bool]  # Whether to process logs by epoch
    custom_cfg: Optional[List[ConfigDict]]  # Custom logging configuration
    num_digits: Optional[int]  # Number of digits for metric formatting


class EnvConfig(TypedDict, total=False):
    """Environment configuration."""
    cudnn_benchmark: Optional[bool]  # Whether to enable cuDNN benchmark mode
    mp_cfg: Optional[ConfigDict]  # Multi-processing configuration
    dist_cfg: Optional[ConfigDict]  # Distributed training configuration


class RuntimeConfig(TypedDict, total=False):
    """Complete runtime configuration."""
    default_scope: Optional[str]  # Default scope for registry ('mmpose')
    default_hooks: Optional[DefaultHooksConfig]  # Default hooks configuration
    custom_hooks: Optional[List[HookConfig]]  # Custom hooks list
    env_cfg: Optional[EnvConfig]  # Environment configuration
    vis_backends: Optional[List[Union[LocalVisBackendConfig, TensorboardVisBackendConfig, WandbVisBackendConfig]]]  # Global visualization backends
    visualizer: Optional[VisualizerConfig]  # Visualizer configuration
    log_processor: Optional[LogProcessorConfig]  # Log processor configuration
    log_level: Optional[Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']]  # Logging level
    load_from: Optional[str]  # Path to checkpoint file to load weights from
    resume: Optional[Union[bool, str]]  # Whether to resume training or path to resume checkpoint
    launcher: Optional[Literal['none', 'pytorch', 'slurm', 'mpi']]  # Distributed launcher type
    work_dir: Optional[str]  # Working directory for outputs
    seed: Optional[int]  # Random seed for reproducibility
    diff_seed: Optional[bool]  # Whether to use different seeds for different ranks
    deterministic: Optional[bool]  # Whether to set deterministic mode for reproducibility