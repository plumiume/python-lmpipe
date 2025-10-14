"""
Sample MMPose configuration with TypedDict support for enhanced IDE experience.

This configuration demonstrates how to use the TypedDict definitions for 
type-safe configuration creation with full Pylance/IDE support.

To enable type checking and IntelliSense:
1. Import the appropriate TypedDict from mmpose_config_typeddict
2. Add type annotation to your config variable
3. Enjoy auto-completion and error detection!
"""

# Enable type checking with TypedDict
from mmpose_config_typeddict import CompleteMMPoseConfig

# Type annotation enables IDE support
config: CompleteMMPoseConfig

# Configuration with full type safety and IntelliSense
config = {
    # Base configuration inheritance
    '_base_': [
        'mmpose_config_typeddict/_base_/default_runtime.py'
    ],
    
    # Model configuration with type checking
    'model': {
        'type': 'TopdownPoseEstimator',  # IDE will validate this value
        'data_preprocessor': {
            'type': 'PoseDataPreprocessor',
            'mean': [123.675, 116.28, 103.53],  # Type: List[float]
            'std': [58.395, 57.12, 57.375],
            'bgr_to_rgb': True  # Type: bool
        },
        'backbone': {
            'type': 'RTMPoseNet',  # Will show available backbone types
            'arch': 'P5',
            'expand_ratio': 0.5,
            'deepen_factor': 0.67,
            'widen_factor': 0.75,
            'out_indices': (4,),
            'channel_attention': True
        },
        'head': {
            'type': 'RTMCCHead',  # IDE auto-completion for head types
            'in_channels': 768,   # Type: int
            'out_channels': 17,   # Number of COCO keypoints
            'input_size': (192, 256),  # Type: Tuple[int, int]
            'in_featuremap_size': (6, 8),
            'simcc_split_ratio': 2.0,  # Type: float
            'final_layer_kernel_size': 7,
            'gau_cfg': {
                'hidden_dims': 256,
                'use_bias': False,
                'act_fn': 'SiLU',
                'use_rel_bias': False,
                'pos_enc': False
            },
            'loss': {
                'type': 'KLDiscretLoss',
                'use_target_weight': True,
                'beta': 10.0,
                'label_softmax': True
            }
        },
        'test_cfg': {
            'flip_test': True,      # IDE will show: bool
            'flip_mode': 'heatmap', # IDE will show: Literal['heatmap', 'regression']
            'shift_heatmap': True,
            'normalize': False
        }
    },
    
    # Training data loader with type validation
    'train_dataloader': {
        'batch_size': 256,  # Type: int (IDE will catch string assignments)
        'num_workers': 8,
        'persistent_workers': True,  # Type: bool
        'sampler': {
            'type': 'DefaultSampler',
            'shuffle': True
        },
        'dataset': {
            'type': 'CocoDataset',  # IDE validates dataset type
            'data_root': 'data/coco/',
            'data_mode': 'topdown',  # Literal type validation
            'ann_file': 'annotations/person_keypoints_train2017.json',
            'data_prefix': {
                'img': 'train2017/'  # IDE understands nested dict structure
            },
            'pipeline': [  # Type: List[TransformConfig]
                {'type': 'LoadImage'},
                {'type': 'GetBBoxCenterScale'},
                {
                    'type': 'RandomFlip',
                    'direction': 'horizontal'  # Literal validation
                },
                {'type': 'RandomHalfBody'},
                {
                    'type': 'RandomBBoxTransform',
                    'scale_factor': [0.6, 1.4],  # Type: Tuple[float, float]
                    'rotate_factor': 80
                },
                {
                    'type': 'TopdownAffine',
                    'input_size': (192, 256)  # Type validation
                },
                {
                    'type': 'GenerateTarget',
                    'encoder': 'MSRAHeatmap'
                },
                {'type': 'PackPoseInputs'}
            ],
            'test_mode': False
        }
    },
    
    # Validation data loader
    'val_dataloader': {
        'batch_size': 32,
        'num_workers': 4,
        'persistent_workers': True,
        'drop_last': False,
        'sampler': {
            'type': 'DefaultSampler',
            'shuffle': False,
            'round_up': False
        },
        'dataset': {
            'type': 'CocoDataset',
            'data_root': 'data/coco/',
            'data_mode': 'topdown',
            'ann_file': 'annotations/person_keypoints_val2017.json',
            'bbox_file': 'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
            'data_prefix': {
                'img': 'val2017/'
            },
            'pipeline': [
                {'type': 'LoadImage'},
                {'type': 'GetBBoxCenterScale'},
                {
                    'type': 'TopdownAffine',
                    'input_size': (192, 256)
                },
                {'type': 'PackPoseInputs'}
            ],
            'test_mode': True
        }
    },
    
    # Optimizer configuration with type safety
    'optim_wrapper': {
        'optimizer': {
            'type': 'Adam',  # IDE shows available optimizer types
            'lr': 0.0005,    # Type: float
            'weight_decay': 0.0001
        },
        'paramwise_cfg': {
            'norm_decay_mult': 0,    # Normalization layer weight decay multiplier
            'bias_decay_mult': 0,    # Bias parameter weight decay multiplier
            'bypass_duplicate': True
        },
        'clip_grad': {
            'max_norm': 35,    # Type: float - Maximum gradient norm
            'norm_type': 2     # Type: float - L2 norm
        }
    },
    
    # Learning rate scheduler
    'param_scheduler': [
        {
            'type': 'LinearLR',  # Linear warmup
            'start_factor': 0.001,
            'by_epoch': False,   # Type: bool
            'begin': 0,
            'end': 1000
        },
        {
            'type': 'MultiStepLR',  # Multi-step decay
            'begin': 0,
            'end': 420,
            'by_epoch': True,
            'milestones': [170, 200],  # Type: List[int]
            'gamma': 0.1  # Type: float
        }
    ],
    
    # Training loop configuration
    'train_cfg': {
        'type': 'EpochBasedTrainLoop',
        'max_epochs': 420,      # Type: int
        'val_interval': 10      # Type: int - Validation every 10 epochs
    },
    
    # Validation loop
    'val_cfg': {
        'type': 'ValLoop'
    },
    
    # Test loop
    'test_cfg': {
        'type': 'TestLoop'
    },
    
    # Evaluation metrics
    'val_evaluator': {
        'type': 'CocoMetric',  # IDE validates evaluator types
        'ann_file': 'data/coco/annotations/person_keypoints_val2017.json'
    },
    
    'test_evaluator': {
        'type': 'CocoMetric',
        'ann_file': 'data/coco/annotations/person_keypoints_val2017.json'
    },
    
    # Runtime configuration
    'work_dir': './work_dirs/rtmpose_typed_config_example',
    'experiment_name': 'rtmpose_with_typeddict',
    
    # Advanced options with type checking
    'auto_scale_lr': {
        'enable': False,       # Type: bool
        'base_batch_size': 512 # Type: int
    },
    
    # Mixed precision training
    'fp16': {
        'loss_scale': 'dynamic'  # Type: Union[str, float]
    },
    
    # PyTorch 2.0 compile
    'compile': True,  # Type: bool
    
    # Reproducibility
    'seed': 42,           # Type: int
    'deterministic': False # Type: bool
}

# IDE Benefits demonstrated:
# 1. Auto-completion for all configuration keys
# 2. Type validation (prevents config['batch_size'] = "invalid")
# 3. Hover documentation for each field
# 4. Navigation to type definitions
# 5. Error highlighting for invalid configurations

# Example of IDE catching errors (uncomment to see):
# config['model']['type'] = 123  # ❌ Type error: Expected str, got int
# config['train_dataloader']['batch_size'] = "32"  # ❌ Type error: Expected int, got str
# config['optim_wrapper']['optimizer']['lr'] = "0.001"  # ❌ Type error: Expected float, got str