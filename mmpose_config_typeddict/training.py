"""
Training configuration TypedDict definitions for MMPose.

Provides type definitions for optimizers, schedulers, training loops, and training-related settings.
"""

from typing import Dict, Any, Union, List, Optional, Literal, TypedDict
from .base import ConfigDict

__all__ = [
    'TrainingConfig', 'OptimWrapperConfig', 'OptimizerConfig', 'ParamSchedulerConfig',
    'TrainCfgConfig', 'ValCfgConfig', 'TestCfgConfig'
]


class OptimizerConfig(TypedDict, total=False):
    """Optimizer configuration."""
    type: str  # Optimizer type ('Adam', 'SGD', 'AdamW', etc.)
    lr: float  # Learning rate
    weight_decay: Optional[float]  # Weight decay (L2 regularization) coefficient
    momentum: Optional[float]  # Momentum factor for SGD-based optimizers
    betas: Optional[tuple[float, float]]  # Beta coefficients for Adam-based optimizers
    eps: Optional[float]  # Epsilon value for numerical stability
    amsgrad: Optional[bool]  # Whether to use AMSGrad variant for Adam


class ParamwiseCfgConfig(TypedDict, total=False):
    """Parameter-wise optimizer configuration."""
    norm_decay_mult: Optional[float]  # Weight decay multiplier for normalization layers
    bias_decay_mult: Optional[float]  # Weight decay multiplier for bias parameters
    bypass_duplicate: Optional[bool]  # Whether to bypass duplicate parameter optimization
    custom_keys: Optional[Dict[str, ConfigDict]]  # Custom parameter-specific settings


class ClipGradConfig(TypedDict, total=False):
    """Gradient clipping configuration."""
    max_norm: float  # Maximum gradient norm for clipping
    norm_type: Optional[float]  # Type of norm to compute (2.0 for L2 norm)
    error_if_nonfinite: Optional[bool]  # Whether to raise error on non-finite gradients


class OptimWrapperConfig(TypedDict, total=False):
    """Optimizer wrapper configuration."""
    type: Optional[str]  # Optimizer wrapper type ('OptimWrapper', 'AmpOptimWrapper')
    optimizer: OptimizerConfig  # Optimizer configuration
    paramwise_cfg: Optional[ParamwiseCfgConfig]  # Parameter-wise optimization settings
    clip_grad: Optional[ClipGradConfig]  # Gradient clipping configuration
    accumulative_counts: Optional[int]  # Number of steps for gradient accumulation
    loss_scale: Optional[Union[str, float]]  # Loss scaling for mixed precision ('dynamic' or float)


class ParamSchedulerConfig(TypedDict, total=False):
    """Parameter scheduler configuration for learning rate scheduling."""
    type: str  # Scheduler type ('LinearLR', 'MultiStepLR', 'CosineAnnealingLR', etc.)
    begin: Optional[int]  # Beginning epoch/iteration for scheduler
    end: Optional[int]  # Ending epoch/iteration for scheduler
    by_epoch: Optional[bool]  # Whether scheduler works by epoch (True) or iteration (False)
    
    # LinearLR specific
    start_factor: Optional[float]  # Starting factor for linear learning rate warmup
    end_factor: Optional[float]  # Ending factor for linear learning rate
    
    # MultiStepLR specific
    milestones: Optional[List[int]]  # Epoch/iteration milestones for learning rate decay
    gamma: Optional[float]  # Multiplicative factor for learning rate decay
    
    # CosineAnnealingLR specific
    T_max: Optional[int]  # Maximum number of iterations for cosine annealing
    eta_min: Optional[float]  # Minimum learning rate
    
    # ExponentialLR specific
    gamma: Optional[float]  # Multiplicative factor for exponential decay
    
    # PolynomialLR specific
    power: Optional[float]  # Power for polynomial learning rate decay
    
    # OneCycleLR specific
    max_lr: Optional[float]  # Maximum learning rate for OneCycle policy
    total_steps: Optional[int]  # Total number of steps for OneCycle
    pct_start: Optional[float]  # Percentage of cycle spent increasing learning rate
    anneal_strategy: Optional[Literal['cos', 'linear']]  # Annealing strategy
    div_factor: Optional[float]  # Initial learning rate divisor
    final_div_factor: Optional[float]  # Final learning rate divisor


class TrainCfgConfig(TypedDict, total=False):
    """Training loop configuration."""
    type: Optional[str]  # Training loop type ('EpochBasedTrainLoop', 'IterBasedTrainLoop')
    max_epochs: Optional[int]  # Maximum number of training epochs
    max_iters: Optional[int]  # Maximum number of training iterations
    val_interval: Optional[int]  # Validation interval in epochs/iterations
    val_begin: Optional[int]  # Epoch/iteration to begin validation
    dynamic_intervals: Optional[List[tuple[int, int]]]  # Dynamic validation intervals


class ValCfgConfig(TypedDict, total=False):
    """Validation loop configuration."""
    type: Optional[str]  # Validation loop type ('ValLoop')
    fp16: Optional[bool]  # Whether to use FP16 for validation


class TestCfgConfig(TypedDict, total=False):
    """Test loop configuration."""
    type: Optional[str]  # Test loop type ('TestLoop')
    fp16: Optional[bool]  # Whether to use FP16 for testing


class AutoScaleLRConfig(TypedDict, total=False):
    """Automatic learning rate scaling configuration."""
    enable: bool  # Whether to enable automatic learning rate scaling
    base_batch_size: int  # Base batch size for learning rate scaling
    


class FP16Config(TypedDict, total=False):
    """Mixed precision training configuration."""
    loss_scale: Optional[Union[str, float]]  # Loss scale ('dynamic' or fixed value)
    init_scale: Optional[float]  # Initial loss scale for dynamic scaling
    growth_factor: Optional[float]  # Growth factor for dynamic loss scaling
    backoff_factor: Optional[float]  # Backoff factor for dynamic loss scaling
    growth_interval: Optional[int]  # Interval for loss scale growth


class TrainingConfig(TypedDict, total=False):
    """Complete training configuration."""
    optim_wrapper: OptimWrapperConfig  # Optimizer wrapper configuration
    param_scheduler: Union[ParamSchedulerConfig, List[ParamSchedulerConfig]]  # Learning rate scheduler(s)
    train_cfg: TrainCfgConfig  # Training loop configuration
    val_cfg: Optional[ValCfgConfig]  # Validation loop configuration
    test_cfg: Optional[TestCfgConfig]  # Test loop configuration
    auto_scale_lr: Optional[AutoScaleLRConfig]  # Automatic learning rate scaling
    fp16: Optional[FP16Config]  # Mixed precision training configuration
    compile: Optional[bool]  # Whether to use PyTorch 2.0 compile
    find_unused_parameters: Optional[bool]  # Whether to find unused parameters in DDP