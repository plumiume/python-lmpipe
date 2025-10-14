"""
Complete MMPose configuration TypedDict definitions.

Provides the main configuration class that combines all component configurations
for a complete MMPose training/inference setup.
"""

from typing import Union, List, Optional, TypedDict
from .base import ConfigDict
from .model import ModelConfig, RTMPoseConfig, ViTPoseConfig
from .dataset import DataLoaderConfig
from .training import TrainingConfig, OptimWrapperConfig, ParamSchedulerConfig, TrainCfgConfig, ValCfgConfig, TestCfgConfig
from .evaluation import EvaluatorConfig
from .runtime import RuntimeConfig, DefaultHooksConfig, VisualizerConfig, LogProcessorConfig, EnvConfig

__all__ = ['CompleteMMPoseConfig', 'MMPoseExperimentConfig']


class CompleteMMPoseConfig(TypedDict, total=False):
    """Complete MMPose configuration combining all components."""
    
    # Base configuration inheritance
    _base_: Optional[List[str]]  # Base configuration files to inherit from
    
    # Model configuration
    model: Union[ModelConfig, RTMPoseConfig, ViTPoseConfig]  # Model architecture and components
    
    # Data configuration
    train_dataloader: Optional[DataLoaderConfig]  # Training data loader configuration
    val_dataloader: Optional[DataLoaderConfig]  # Validation data loader configuration
    test_dataloader: Optional[DataLoaderConfig]  # Test data loader configuration
    
    # Training configuration
    optim_wrapper: Optional[OptimWrapperConfig]  # Optimizer wrapper configuration
    param_scheduler: Optional[Union[ParamSchedulerConfig, List[ParamSchedulerConfig]]]  # Learning rate scheduler(s)
    train_cfg: Optional[TrainCfgConfig]  # Training loop configuration
    val_cfg: Optional[ValCfgConfig]  # Validation loop configuration
    test_cfg: Optional[TestCfgConfig]  # Test loop configuration
    
    # Evaluation configuration
    val_evaluator: Optional[Union[EvaluatorConfig, List[EvaluatorConfig]]]  # Validation evaluator(s)
    test_evaluator: Optional[Union[EvaluatorConfig, List[EvaluatorConfig]]]  # Test evaluator(s)
    
    # Runtime configuration
    default_scope: Optional[str]  # Default scope for registry ('mmpose')
    default_hooks: Optional[DefaultHooksConfig]  # Default hooks configuration
    custom_hooks: Optional[List[dict]]  # Custom hooks list
    env_cfg: Optional[EnvConfig]  # Environment configuration
    vis_backends: Optional[List[dict]]  # Visualization backends
    visualizer: Optional[VisualizerConfig]  # Visualizer configuration
    log_processor: Optional[LogProcessorConfig]  # Log processor configuration
    
    # Experiment configuration
    work_dir: Optional[str]  # Working directory for experiment outputs
    experiment_name: Optional[str]  # Name of the experiment
    load_from: Optional[str]  # Path to checkpoint file to load weights from
    resume: Optional[Union[bool, str]]  # Whether to resume training or path to resume checkpoint
    
    # Advanced training options
    auto_scale_lr: Optional[dict]  # Automatic learning rate scaling configuration
    fp16: Optional[dict]  # Mixed precision training configuration
    compile: Optional[bool]  # Whether to use PyTorch 2.0 compile
    
    # Reproducibility
    seed: Optional[int]  # Random seed for reproducibility
    diff_seed: Optional[bool]  # Whether to use different seeds for different ranks
    deterministic: Optional[bool]  # Whether to set deterministic mode
    
    # Logging and monitoring
    log_level: Optional[str]  # Logging level ('INFO', 'DEBUG', etc.)
    launcher: Optional[str]  # Distributed launcher type ('pytorch', 'slurm', etc.)


class MMPoseExperimentConfig(CompleteMMPoseConfig):
    """Extended configuration for MMPose experiments with additional metadata."""
    
    # Experiment metadata
    experiment_name: str  # Required experiment name
    description: Optional[str]  # Experiment description
    tags: Optional[List[str]]  # Tags for experiment categorization
    notes: Optional[str]  # Additional notes about the experiment
    
    # Dataset metadata
    dataset_name: Optional[str]  # Name of the dataset being used
    dataset_version: Optional[str]  # Version of the dataset
    num_classes: Optional[int]  # Number of keypoint classes
    
    # Model metadata  
    model_name: Optional[str]  # Name/identifier of the model
    model_version: Optional[str]  # Version of the model architecture
    pretrained: Optional[str]  # Path or identifier of pretrained weights
    
    # Hardware configuration
    gpus: Optional[int]  # Number of GPUs to use
    gpu_ids: Optional[List[int]]  # Specific GPU IDs to use
    workers_per_gpu: Optional[int]  # Number of workers per GPU
    
    # Performance tracking
    expected_metrics: Optional[dict]  # Expected performance metrics
    baseline_metrics: Optional[dict]  # Baseline metrics for comparison
    
    # Resource constraints
    max_memory_gb: Optional[float]  # Maximum memory usage in GB
    max_time_hours: Optional[float]  # Maximum training time in hours