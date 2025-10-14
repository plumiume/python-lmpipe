"""
Example usage of MMPose configuration TypedDict definitions.

This module provides practical examples of how to use the TypedDict definitions
for creating type-safe MMPose configurations with excellent IDE support.
"""

from mmpose_config_typeddict import CompleteMMPoseConfig, RTMPoseConfig, CocoDatasetConfig

# Example: RTMPose configuration with type safety
def create_rtmpose_config() -> CompleteMMPoseConfig:
    """Create a complete RTMPose configuration with full type checking."""
    
    config: CompleteMMPoseConfig = {
        '_base_': [
            'configs/_base_/default_runtime.py'
        ],
        
        'model': {
            'type': 'TopdownPoseEstimator',
            'data_preprocessor': {
                'type': 'PoseDataPreprocessor',
                'mean': [123.675, 116.28, 103.53],
                'std': [58.395, 57.12, 57.375],
                'bgr_to_rgb': True
            },
            'backbone': {
                'type': 'CSPNeXt',
                'arch': 'P5',
                'expand_ratio': 0.5,
                'deepen_factor': 0.67,
                'widen_factor': 0.75,
                'out_indices': (4,),
                'channel_attention': True
            },
            'head': {
                'type': 'RTMCCHead',
                'in_channels': 768,
                'out_channels': 17,
                'input_size': (192, 256),
                'in_featuremap_size': (6, 8),
                'simcc_split_ratio': 2.0,
                'final_layer_kernel_size': 7
            },
            'test_cfg': {
                'flip_test': True,
                'flip_mode': 'heatmap',
                'shift_heatmap': True
            }
        },
        
        'train_dataloader': {
            'batch_size': 256,
            'num_workers': 8,
            'persistent_workers': True,
            'dataset': {
                'type': 'CocoDataset',
                'data_root': 'data/coco/',
                'data_mode': 'topdown',
                'ann_file': 'annotations/person_keypoints_train2017.json',
                'data_prefix': {'img': 'train2017/'},
                'pipeline': [
                    {'type': 'LoadImage'},
                    {'type': 'GetBBoxCenterScale'},
                    {'type': 'RandomFlip', 'direction': 'horizontal'},
                    {'type': 'TopdownAffine', 'input_size': (192, 256)},
                    {'type': 'GenerateTarget', 'encoder': 'MSRAHeatmap'},
                    {'type': 'PackPoseInputs'}
                ]
            }
        },
        
        'val_dataloader': {
            'batch_size': 32,
            'num_workers': 4,
            'dataset': {
                'type': 'CocoDataset',
                'data_root': 'data/coco/',
                'data_mode': 'topdown',
                'ann_file': 'annotations/person_keypoints_val2017.json',
                'data_prefix': {'img': 'val2017/'},
                'pipeline': [
                    {'type': 'LoadImage'},
                    {'type': 'GetBBoxCenterScale'},
                    {'type': 'TopdownAffine', 'input_size': (192, 256)},
                    {'type': 'PackPoseInputs'}
                ]
            }
        },
        
        'optim_wrapper': {
            'optimizer': {
                'type': 'Adam',
                'lr': 0.0005,
                'weight_decay': 0.0001
            }
        },
        
        'param_scheduler': [
            {
                'type': 'LinearLR',
                'start_factor': 0.001,
                'by_epoch': False,
                'begin': 0,
                'end': 1000
            },
            {
                'type': 'MultiStepLR',
                'begin': 0,
                'end': 420,
                'by_epoch': True,
                'milestones': [170, 200],
                'gamma': 0.1
            }
        ],
        
        'train_cfg': {
            'type': 'EpochBasedTrainLoop',
            'max_epochs': 420,
            'val_interval': 10
        },
        
        'val_evaluator': {
            'type': 'CocoMetric',
            'ann_file': 'data/coco/annotations/person_keypoints_val2017.json'
        },
        
        'work_dir': './work_dirs/rtmpose_example'
    }
    
    return config


def create_single_person_optimized_config() -> CompleteMMPoseConfig:
    """Create an optimized configuration for single person pose estimation."""
    
    config: CompleteMMPoseConfig = {
        'model': {
            'type': 'TopdownPoseEstimator',
            'test_cfg': {
                'flip_test': False,  # Disable flip test for speed
                'shift_heatmap': False  # Disable shift for speed
            }
        },
        
        'train_dataloader': {
            'batch_size': 64,  # Smaller batch for single person
            'num_workers': 4,
            'dataset': {
                'type': 'CocoDataset',
                'pipeline': [
                    {'type': 'LoadImage'},
                    {'type': 'GetBBoxCenterScale'},
                    {'type': 'TopdownAffine', 'input_size': (192, 256)},
                    {'type': 'PackPoseInputs'}
                ]
            }
        },
        
        'optim_wrapper': {
            'optimizer': {
                'type': 'Adam',
                'lr': 0.001,  # Higher learning rate for faster convergence
                'weight_decay': 0.0001
            }
        }
    }
    
    return config


# Usage example with type checking
if __name__ == "__main__":
    # Create configuration with full type safety
    config = create_rtmpose_config()
    
    # IDE will provide auto-completion and type checking
    print(f"Model type: {config['model']['type']}")
    print(f"Batch size: {config['train_dataloader']['batch_size']}")
    print(f"Learning rate: {config['optim_wrapper']['optimizer']['lr']}")
    
    # This will cause a type error if you try to assign wrong types:
    # config['model']['type'] = 123  # Type error!
    # config['train_dataloader']['batch_size'] = "invalid"  # Type error!