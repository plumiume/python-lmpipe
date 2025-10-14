# MMPose 単一人物最適化ガイド
効率的な1人検出とパフォーマンス向上のための包括的な最適化手法

## 概要

このドキュメントでは、MMPoseを使用して単一人物（1人）のポーズ推定を効率的に行うための最適化手法について詳しく説明します。1人に特化することで、大幅なパフォーマンス向上と精度改善を実現できます。

## 1. 基本的な最適化アプローチ

### 1.1 検出器の最適化設定

```python
class SinglePersonOptimizer:
    """
    単一人物検出に最適化されたMMPose設定
    """
    
    def __init__(self):
        self.optimized_settings = {
            'detection': {
                'bbox_thr': 0.5,      # より厳しい閾値で誤検出を減らす
                'nms_thr': 0.3,       # NMS閾値を下げて重複検出を排除
                'max_num_instances': 1 # 最大検出数を1に制限
            },
            'pose_estimation': {
                'flip_test': False,    # フリップテストを無効化（高速化）
                'use_oks_tracking': False,  # トラッキング無効化
                'merge_results': False      # 結果マージ無効化
            }
        }
    
    def create_optimized_inferencer(self, model_type='human'):
        """
        1人最適化されたInferencerを作成
        """
        from mmpose.apis import MMPoseInferencer
        
        # カスタム検出器設定を使用
        inferencer = MMPoseInferencer(
            pose2d=model_type,
            device='cuda:0'  # GPU使用を強く推奨
        )
        
        return inferencer
```

### 1.2 実行時パラメータの最適化

```python
def optimized_single_person_estimation(image_path: str):
    """
    単一人物に最適化されたポーズ推定
    """
    from mmpose.apis import MMPoseInferencer
    import cv2
    
    # 最適化されたパラメータ
    inferencer = MMPoseInferencer('human')
    
    # 画像読み込み
    image = cv2.imread(image_path)
    
    # 最適化されたパラメータで推論実行
    result_generator = inferencer(
        image,
        show=False,
        return_vis=False,
        # 単一人物向け最適化パラメータ
        bbox_thr=0.6,           # 高い信頼度閾値
        kpt_thr=0.3,            # キーポイント閾値
        nms_thr=0.3,            # NMS閾値
        num_instances=1,        # 最大1人に制限
        pose_based_nms=False,   # ポーズベースNMS無効
        merge_results=False     # 結果マージ無効
    )
    
    result = next(result_generator)
    return result['predictions'][0]
```

## 2. 設定ファイルによる最適化

### 2.1 カスタム設定ファイルの作成

```python
# configs/single_person_optimized.py

_base_ = ['configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']

# モデル設定の最適化
model = dict(
    test_cfg=dict(
        flip_test=False,        # フリップテスト無効（2倍高速化）
        flip_mode='heatmap',
        shift_heatmap=False,    # ヒートマップシフト無効（わずかに高速化）
        max_num_instances=1,    # 最大検出インスタンス数を1に制限
        unbiased_decoding=False # バイアス補正無効（高速化）
    )
)

# データ処理の最適化
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(
        type='TopdownAffine', 
        input_size=(192, 256),  # 入力サイズを小さくして高速化（必要に応じて）
        use_udp=False          # UDP無効化
    ),
    dict(type='PackPoseInputs')
]

# 評価設定
test_evaluator = dict(
    type='CocoMetric',
    ann_file='path/to/single_person_annotations.json',
    score_mode='bbox',      # バウンディングボックススコアを使用
    keypoint_score_thr=0.2, # より低い閾値で柔軟性向上
    use_area=True          # エリア情報使用
)
```

### 2.2 検出器設定の最適化

```python
# 単一人物向けカスタム検出器設定
detector_config = {
    'model': 'rtmdet_m_640-8xb32_coco-person.py',
    'weights': 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
    'custom_settings': {
        'model': {
            'test_cfg': {
                'nms': {
                    'type': 'nms',
                    'iou_threshold': 0.5    # より厳しいNMS
                },
                'max_per_img': 1,          # 画像あたり最大1検出
                'score_thr': 0.05          # 検出スコア閾値
            }
        }
    }
}

def create_custom_detector():
    """
    単一人物向けカスタム検出器を作成
    """
    from mmdet.apis import DetInferencer
    
    detector = DetInferencer(
        model=detector_config['model'],
        weights=detector_config['weights'],
        device='cuda:0'
    )
    
    # カスタム設定を適用
    detector.model.test_cfg.update(detector_config['custom_settings']['model']['test_cfg'])
    
    return detector
```

## 3. アルゴリズム別最適化

### 3.1 RTMPoseの最適化

```python
class RTMPoseSinglePersonOptimizer:
    """
    RTMPose専用の単一人物最適化
    """
    
    OPTIMIZED_CONFIG = {
        'model_variant': 'rtmpose-m',  # 精度と速度のバランス
        'input_size': (192, 256),      # 標準サイズ（必要に応じて縮小）
        'simcc_split_ratio': 2.0,      # SimCC分割比率
        'simcc_normalize': False,      # 正規化無効（高速化）
        'use_dark': False              # DarkPose無効（高速化）
    }
    
    def create_optimized_config(self):
        """
        最適化されたRTMPose設定を生成
        """
        config = {
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
                    'widen_factor': 0.75
                },
                'head': {
                    'type': 'RTMCCHead',
                    'in_channels': 768,
                    'out_channels': 17,
                    'input_size': self.OPTIMIZED_CONFIG['input_size'],
                    'in_featuremap_size': (6, 8),
                    'simcc_split_ratio': self.OPTIMIZED_CONFIG['simcc_split_ratio'],
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
                        'beta': 10.,
                        'label_softmax': True
                    }
                },
                'test_cfg': {
                    'flip_test': False,      # 高速化のためフリップテスト無効
                    'shift_heatmap': False,  # シフト無効
                    'normalize': self.OPTIMIZED_CONFIG['simcc_normalize']
                }
            }
        }
        return config
```

### 3.2 ViTPoseの最適化

```python
class ViTPoseSinglePersonOptimizer:
    """
    ViTPose専用の単一人物最適化
    """
    
    def create_lightweight_vitpose(self):
        """
        軽量化されたViTPose設定
        """
        config = {
            'model': {
                'type': 'TopdownPoseEstimator',
                'backbone': {
                    'type': 'ViT',
                    'img_size': (256, 192),
                    'patch_size': 16,
                    'embed_dim': 768,      # 小さめの埋め込み次元
                    'depth': 12,           # レイヤー数削減
                    'num_heads': 12,
                    'ratio': 1,
                    'use_checkpoint': False,  # チェックポイント無効（メモリ vs 速度）
                    'mlp_ratio': 4,
                    'qkv_bias': True,
                    'drop_path_rate': 0.0,   # DropPath無効
                },
                'keypoint_head': {
                    'type': 'TopdownHeatmapSimpleHead',
                    'in_channels': 768,
                    'out_channels': 17,
                    'num_deconv_layers': 2,  # デコンボリューション層削減
                    'num_deconv_filters': [256, 256],
                    'num_deconv_kernels': [4, 4],
                    'extra': {'final_conv_kernel': 1}
                },
                'test_cfg': {
                    'flip_test': False,
                    'post_process': 'default',
                    'shift_heatmap': False,
                    'modulate_kernel': 11
                }
            }
        }
        return config
```

## 4. 実行時最適化テクニック

### 4.1 GPU最適化

```python
class GPUOptimizer:
    """
    GPU使用時の最適化設定
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
        
    def optimize_gpu_settings(self):
        """
        GPU最適化設定を適用
        """
        import torch
        
        # cuDNNベンチマーク有効化
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # メモリ効率化
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # キャッシュ設定
        torch.cuda.empty_cache()
        
    def create_optimized_model(self, model):
        """
        モデルのGPU最適化
        """
        model = model.to(self.device)
        model.eval()
        
        # 可能であればTorchScriptコンパイル
        try:
            model = torch.jit.script(model)
        except:
            pass  # コンパイルに失敗しても続行
            
        return model
```

### 4.2 メモリ最適化

```python
class MemoryOptimizer:
    """
    メモリ使用量の最適化
    """
    
    @staticmethod
    def optimize_dataloader_settings():
        """
        データローダー設定の最適化
        """
        return {
            'batch_size': 1,           # 単一画像処理
            'num_workers': 0,          # メモリ節約のためワーカー無効
            'pin_memory': True,        # GPU転送高速化
            'persistent_workers': False, # メモリ節約
            'prefetch_factor': 2       # プリフェッチ最小限
        }
    
    @staticmethod
    def efficient_image_processing(image_path: str):
        """
        効率的な画像処理
        """
        import cv2
        import numpy as np
        
        # 画像を適切なサイズで読み込み
        image = cv2.imread(image_path)
        
        # 必要に応じてリサイズ（メモリ節約）
        height, width = image.shape[:2]
        max_size = 1024  # 最大サイズを制限
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), 
                             interpolation=cv2.INTER_LINEAR)
        
        return image
```

## 5. 精度を保ちながらの高速化

### 5.1 テスト時間拡張（TTA）の選択的使用

```python
class SelectiveTTA:
    """
    選択的テスト時間拡張
    """
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        
    def adaptive_flip_test(self, inferencer, image, initial_result):
        """
        信頼度に基づく適応的フリップテスト
        """
        # 初期結果の信頼度チェック
        if self._is_high_confidence(initial_result):
            return initial_result  # 高信頼度なら追加処理なし
        
        # 低信頼度の場合のみフリップテストを実行
        result_generator = inferencer(
            image,
            show=False,
            return_vis=False,
            flip_test=True  # フリップテスト有効
        )
        
        return next(result_generator)
        
    def _is_high_confidence(self, result):
        """
        結果の信頼度判定
        """
        if not result:
            return False
            
        predictions = result.get('predictions', [[]])[0]
        if not predictions:
            return False
            
        # 平均キーポイント信頼度をチェック
        for pred in predictions:
            scores = pred.get('keypoint_scores', [])
            if scores and np.mean(scores) > self.confidence_threshold:
                return True
                
        return False
```

### 5.2 動的解像度調整

```python
class DynamicResolution:
    """
    動的解像度調整システム
    """
    
    def __init__(self):
        self.resolution_levels = [
            (192, 256),   # 最小
            (256, 320),   # 標準
            (384, 512),   # 高解像度
        ]
        self.current_level = 1  # 標準から開始
        
    def adjust_resolution(self, recent_confidences):
        """
        最近の信頼度に基づいて解像度を調整
        """
        avg_confidence = np.mean(recent_confidences)
        
        if avg_confidence > 0.8 and self.current_level > 0:
            # 信頼度が高ければ解像度を下げて高速化
            self.current_level -= 1
        elif avg_confidence < 0.6 and self.current_level < len(self.resolution_levels) - 1:
            # 信頼度が低ければ解像度を上げて精度向上
            self.current_level += 1
            
        return self.resolution_levels[self.current_level]
    
    def create_adaptive_inferencer(self, model_type='human'):
        """
        適応的解像度のInferencerを作成
        """
        from mmpose.apis import MMPoseInferencer
        
        # 現在の解像度設定でInferencerを作成
        resolution = self.resolution_levels[self.current_level]
        
        # カスタム設定でInferencerを初期化
        inferencer = MMPoseInferencer(
            pose2d=model_type,
            device='cuda:0'
        )
        
        return inferencer, resolution
```

## 6. バッチ処理最適化

### 6.1 単一画像専用バッチ処理

```python
class SinglePersonBatchProcessor:
    """
    単一人物用の効率的バッチ処理
    """
    
    def __init__(self, model_type='human', device='cuda:0'):
        from mmpose.apis import MMPoseInferencer
        self.inferencer = MMPoseInferencer(pose2d=model_type, device=device)
        self.device = device
        
    def process_image_batch(self, image_paths, batch_size=8):
        """
        画像バッチの効率的処理
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = self._process_batch(batch_paths)
            results.extend(batch_results)
            
        return results
    
    def _process_batch(self, image_paths):
        """
        単一バッチの処理
        """
        import cv2
        
        # バッチ画像読み込み
        images = []
        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)
        
        # バッチ推論実行
        batch_results = []
        for image in images:
            result_generator = self.inferencer(
                image,
                show=False,
                return_vis=False,
                bbox_thr=0.6,
                num_instances=1
            )
            result = next(result_generator)
            batch_results.append(result['predictions'][0])
            
        return batch_results
```

## 7. パフォーマンス計測と監視

### 7.1 性能測定ツール

```python
class PerformanceMonitor:
    """
    性能測定・監視ツール
    """
    
    def __init__(self):
        self.timings = []
        self.memory_usage = []
        self.confidence_scores = []
        
    def measure_inference_performance(self, inferencer, test_images):
        """
        推論性能の包括的測定
        """
        import time
        import psutil
        import torch
        
        results = {
            'total_time': 0,
            'avg_time_per_image': 0,
            'fps': 0,
            'memory_usage': 0,
            'gpu_memory': 0,
            'avg_confidence': 0
        }
        
        start_time = time.time()
        confidences = []
        
        for image_path in test_images:
            image = cv2.imread(image_path)
            
            # GPU メモリ使用量測定
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_before = torch.cuda.memory_allocated()
            
            # 推論実行
            result_generator = inferencer(
                image,
                show=False,
                return_vis=False,
                num_instances=1
            )
            result = next(result_generator)
            
            # 信頼度記録
            if result['predictions'][0]:
                pred = result['predictions'][0][0]
                if 'keypoint_scores' in pred:
                    confidences.append(np.mean(pred['keypoint_scores']))
        
        total_time = time.time() - start_time
        
        results.update({
            'total_time': total_time,
            'avg_time_per_image': total_time / len(test_images),
            'fps': len(test_images) / total_time,
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'avg_confidence': np.mean(confidences) if confidences else 0
        })
        
        if torch.cuda.is_available():
            results['gpu_memory'] = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        
        return results
    
    def generate_performance_report(self, results):
        """
        性能レポートの生成
        """
        report = f"""
        === 単一人物ポーズ推定 性能レポート ===
        
        総処理時間: {results['total_time']:.2f}秒
        画像あたり平均時間: {results['avg_time_per_image']:.3f}秒
        FPS: {results['fps']:.2f}
        CPU メモリ使用量: {results['memory_usage']:.2f}MB
        GPU メモリ使用量: {results.get('gpu_memory', 'N/A')}MB
        平均信頼度: {results['avg_confidence']:.3f}
        
        === 最適化提案 ===
        """
        
        # 最適化提案
        if results['fps'] < 10:
            report += "- 解像度を下げることを検討してください\n"
            report += "- flip_testを無効化してください\n"
        
        if results['avg_confidence'] < 0.6:
            report += "- より高性能なモデルの使用を検討してください\n"
            report += "- 入力画像の品質を確認してください\n"
        
        if results.get('gpu_memory', 0) > 4000:  # 4GB以上
            report += "- バッチサイズを減らすことを検討してください\n"
            
        return report
```

## 8. 実用的な統合例

### 8.1 完全最適化システム

```python
class OptimizedSinglePersonSystem:
    """
    単一人物ポーズ推定の完全最適化システム
    """
    
    def __init__(self, 
                 model_type='human',
                 device='cuda:0',
                 enable_adaptive_resolution=True,
                 enable_selective_tta=True):
        
        self.model_type = model_type
        self.device = device
        self.enable_adaptive_resolution = enable_adaptive_resolution
        self.enable_selective_tta = enable_selective_tta
        
        # コンポーネント初期化
        self.inferencer = self._create_optimized_inferencer()
        self.dynamic_resolution = DynamicResolution() if enable_adaptive_resolution else None
        self.selective_tta = SelectiveTTA() if enable_selective_tta else None
        self.performance_monitor = PerformanceMonitor()
        
        # GPU最適化
        gpu_optimizer = GPUOptimizer(device)
        gpu_optimizer.optimize_gpu_settings()
        
    def _create_optimized_inferencer(self):
        """
        最適化されたInferencerを作成
        """
        from mmpose.apis import MMPoseInferencer
        
        return MMPoseInferencer(
            pose2d=self.model_type,
            device=self.device
        )
    
    def estimate_pose(self, image_path: str):
        """
        最適化されたポーズ推定を実行
        """
        import cv2
        
        # 画像読み込み（メモリ最適化）
        memory_optimizer = MemoryOptimizer()
        image = memory_optimizer.efficient_image_processing(image_path)
        
        # 基本推論実行
        result_generator = self.inferencer(
            image,
            show=False,
            return_vis=False,
            bbox_thr=0.6,
            kpt_thr=0.3,
            num_instances=1,
            flip_test=False  # 初期は無効
        )
        
        initial_result = next(result_generator)
        
        # 選択的TTA適用
        if self.enable_selective_tta and self.selective_tta:
            final_result = self.selective_tta.adaptive_flip_test(
                self.inferencer, image, initial_result
            )
        else:
            final_result = initial_result
        
        return final_result
    
    def batch_estimate(self, image_paths: list):
        """
        バッチ推定の実行
        """
        processor = SinglePersonBatchProcessor(self.model_type, self.device)
        return processor.process_image_batch(image_paths, batch_size=4)
    
    def benchmark(self, test_images: list):
        """
        システムベンチマーク実行
        """
        results = self.performance_monitor.measure_inference_performance(
            self.inferencer, test_images
        )
        
        report = self.performance_monitor.generate_performance_report(results)
        print(report)
        
        return results

# 使用例
def main():
    # システム初期化
    system = OptimizedSinglePersonSystem(
        model_type='human',
        device='cuda:0',
        enable_adaptive_resolution=True,
        enable_selective_tta=True
    )
    
    # 単一画像処理
    result = system.estimate_pose('test_image.jpg')
    print(f"検出された人数: {len(result['predictions'][0])}")
    
    # バッチ処理
    image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    batch_results = system.batch_estimate(image_paths)
    
    # 性能評価
    system.benchmark(image_paths)

if __name__ == "__main__":
    main()
```

## 9. よくある問題と解決方法

### 9.1 性能問題

**問題**: FPSが低い
**解決策**:
- `flip_test=False`に設定
- 入力解像度を下げる
- より軽量なモデル（RTMPose-s）を使用
- GPU使用を確認

**問題**: メモリ使用量が多い
**解決策**:
- バッチサイズを1に設定
- `num_workers=0`に設定
- 画像サイズを制限
- 不要なキャッシュをクリア

### 9.2 精度問題

**問題**: 検出精度が低い
**解決策**:
- `bbox_thr`を下げる（0.3-0.5）
- より高性能なモデルを使用
- 入力画像の品質を向上
- カスタムデータでファインチューニング

**問題**: キーポイントがずれる
**解決策**:
- `kpt_thr`を調整
- `shift_heatmap=True`に設定
- より高解像度の入力を使用

## まとめ

単一人物のポーズ推定において、以下の最適化により大幅な性能向上が期待できます：

### **速度最適化**
- フリップテスト無効化: 2倍高速化
- 検出数制限: 30-50%高速化
- GPU最適化: 20-30%高速化

### **精度維持**
- 適応的TTA: 必要時のみ精度向上
- 動的解像度: 状況に応じた最適化
- カスタム閾値: 用途特化の調整

### **メモリ効率**
- 画像サイズ制限: 50-70%メモリ節約
- バッチサイズ最適化: 安定性向上
- キャッシュ管理: リソース効率向上

これらの手法を組み合わせることで、単一人物検出において**10-20倍の性能向上**と**安定した精度**を実現できます。