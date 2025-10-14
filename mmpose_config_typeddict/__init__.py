"""
MMPose Configuration TypedDict definitions for enhanced IDE support and type safety.

This package provides TypedDict classes for all major MMPose configuration components,
enabling better IDE support, type checking, and documentation for configuration files.
"""

from .base import *
from .model import *
from .dataset import *
from .training import *
from .evaluation import *
from .runtime import *
from .complete_config import *

__all__ = [
    # Base types
    'BaseConfig',
    'ConfigDict',
    
    # Model configuration
    'ModelConfig',
    'BackboneConfig',
    'HeadConfig',
    'DataPreprocessorConfig',
    'TestConfig',
    'RTMPoseConfig',
    'ViTPoseConfig',
    'RTMCCHeadConfig',
    'TopdownHeatmapSimpleHeadConfig',
    
    # Dataset configuration
    'DatasetConfig',
    'DataLoaderConfig',
    'CocoDatasetConfig',
    'PipelineConfig',
    'TransformConfig',
    
    # Training configuration
    'TrainingConfig',
    'OptimWrapperConfig',
    'OptimizerConfig',
    'ParamSchedulerConfig',
    'TrainCfgConfig',
    'ValCfgConfig',
    'TestCfgConfig',
    
    # Evaluation configuration
    'EvaluatorConfig',
    'CocoMetricConfig',
    'PCKAccuracyConfig',
    'AUCConfig',
    'EPEConfig',
    
    # Runtime configuration
    'RuntimeConfig',
    'DefaultHooksConfig',
    'HookConfig',
    'VisualizerConfig',
    'LogProcessorConfig',
    'EnvConfig',
    
    # Complete configuration
    'CompleteMMPoseConfig',
    'MMPoseExperimentConfig'
]