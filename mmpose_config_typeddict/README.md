# MMPose Configuration TypedDict

MMPoseの設定ファイル用のTypeScript風型定義を提供するパッケージです。Pylanceによる型推論とIntelliSenseを有効にし、設定ファイル作成時の生産性と安全性を向上させます。

## 📁 パッケージ構成

```
mmpose_config_typeddict/
├── __init__.py           # パッケージエントリーポイント
├── base.py              # 基本型定義
├── model.py             # モデル設定用TypedDict
├── dataset.py           # データセット設定用TypedDict
├── training.py          # 訓練設定用TypedDict
├── evaluation.py        # 評価設定用TypedDict
├── runtime.py           # ランタイム設定用TypedDict
├── complete_config.py   # 完全設定用TypedDict
├── example_usage.py     # 使用例
└── README.md           # このファイル
```

## 🚀 主な特徴

### ✅ 型安全性
- 設定値の型チェック
- 不正な設定項目の検出
- コンパイル時エラーでランタイムエラーを防止

### 💡 IntelliSense支援
- 設定項目の自動補完
- ドキュメント付きホバー情報
- 設定構造のナビゲーション

### 📚 包括的な型定義
- **モデル設定**: RTMPose, ViTPose, ResNet対応
- **データセット**: COCO, カスタムデータセット対応
- **訓練**: オプティマイザー、スケジューラー設定
- **評価**: 各種メトリクス設定
- **ランタイム**: フック、可視化設定

## 📖 使用方法

### 1. 基本的な使用例

```python
from mmpose_config_typeddict import CompleteMMPoseConfig

# 型安全な設定作成
config: CompleteMMPoseConfig = {
    'model': {
        'type': 'TopdownPoseEstimator',
        'backbone': {
            'type': 'ResNet',
            'depth': 50
        },
        'head': {
            'type': 'TopdownHeatmapSimpleHead',
            'in_channels': 2048,
            'out_channels': 17
        }
    },
    'train_dataloader': {
        'batch_size': 32,
        'dataset': {
            'type': 'CocoDataset',
            'data_root': 'data/coco/'
        }
    }
}
```

### 2. RTMPose設定例

```python
from mmpose_config_typeddict import RTMPoseConfig

rtm_config: RTMPoseConfig = {
    'type': 'TopdownPoseEstimator',
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
        'input_size': (192, 256),
        'simcc_split_ratio': 2.0
    }
}
```

### 3. データセット設定例

```python
from mmpose_config_typeddict import CocoDatasetConfig, DataLoaderConfig

dataset: CocoDatasetConfig = {
    'type': 'CocoDataset',
    'data_root': 'data/coco/',
    'data_mode': 'topdown',
    'ann_file': 'annotations/person_keypoints_train2017.json',
    'pipeline': [
        {'type': 'LoadImage'},
        {'type': 'GetBBoxCenterScale'},
        {'type': 'RandomFlip', 'direction': 'horizontal'},
        {'type': 'TopdownAffine', 'input_size': (192, 256)},
        {'type': 'PackPoseInputs'}
    ]
}

dataloader: DataLoaderConfig = {
    'batch_size': 256,
    'num_workers': 8,
    'persistent_workers': True,
    'dataset': dataset
}
```

### 4. 訓練設定例

```python
from mmpose_config_typeddict import OptimWrapperConfig, ParamSchedulerConfig

optim_config: OptimWrapperConfig = {
    'optimizer': {
        'type': 'Adam',
        'lr': 0.0005,
        'weight_decay': 0.0001
    },
    'paramwise_cfg': {
        'norm_decay_mult': 0,
        'bias_decay_mult': 0
    },
    'clip_grad': {
        'max_norm': 35,
        'norm_type': 2
    }
}

scheduler_config: ParamSchedulerConfig = {
    'type': 'MultiStepLR',
    'by_epoch': True,
    'milestones': [170, 200],
    'gamma': 0.1
}
```

## 🎯 TypedDict クラス一覧

### 🔧 基本型
- `BaseConfig`: 全設定の基底クラス
- `ConfigDict`: 汎用設定辞書型

### 🤖 モデル設定
- `ModelConfig`: 基本モデル設定
- `RTMPoseConfig`: RTMPose専用設定
- `ViTPoseConfig`: ViTPose専用設定
- `BackboneConfig`: バックボーン設定
- `HeadConfig`: ヘッド設定
- `DataPreprocessorConfig`: 前処理設定

### 📊 データ設定
- `DatasetConfig`: 基本データセット設定
- `CocoDatasetConfig`: COCOデータセット設定
- `DataLoaderConfig`: データローダー設定
- `PipelineConfig`: データ処理パイプライン設定

### 🎓 訓練設定
- `TrainingConfig`: 完全訓練設定
- `OptimWrapperConfig`: オプティマイザーラッパー設定
- `OptimizerConfig`: オプティマイザー設定
- `ParamSchedulerConfig`: 学習率スケジューラー設定

### 📈 評価設定
- `CocoMetricConfig`: COCO評価メトリック設定
- `PCKAccuracyConfig`: PCK精度メトリック設定
- `AUCConfig`: AUCメトリック設定
- `EPEConfig`: EPEメトリック設定

### ⚙️ ランタイム設定
- `RuntimeConfig`: 完全ランタイム設定
- `DefaultHooksConfig`: デフォルトフック設定
- `VisualizerConfig`: 可視化設定
- `LogProcessorConfig`: ログ処理設定

### 🎯 統合設定
- `CompleteMMPoseConfig`: 完全MMPose設定
- `MMPoseExperimentConfig`: 実験用拡張設定

## 💡 VS Code / Pylance 設定のコツ

### 1. 設定ファイルの先頭に型アノテーション

```python
# config/my_config.py
from mmpose_config_typeddict import CompleteMMPoseConfig

# 型アノテーションを追加してIntelliSenseを有効化
config: CompleteMMPoseConfig
config = dict(
    # ここで自動補完とエラーチェックが働く
    model=dict(
        type='TopdownPoseEstimator',  # 正しい値のみ受け入れ
        # type='InvalidType',         # エラーになる
    )
)
```

### 2. 設定検証関数の作成

```python
def validate_config(config: CompleteMMPoseConfig) -> bool:
    """設定の妥当性を型レベルでチェック"""
    # この関数内では完全な型チェックが有効
    required_fields = ['model']
    
    for field in required_fields:
        if field not in config:
            return False
    
    return True
```

### 3. VS Code設定の推奨項目

```json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.completeFunctionParens": true,
    "editor.quickSuggestions": {
        "strings": true
    }
}
```

## 🎨 利点とメリット

### 開発効率の向上
- **自動補完**: 設定項目名の入力支援
- **エラー検出**: タイポや型ミスの即座発見
- **ドキュメント**: ホバーでパラメータ説明表示

### 設定品質の向上
- **一貫性**: 標準化された設定構造
- **検証**: 型レベルでの設定検証
- **保守性**: 構造化された設定管理

### チーム開発の促進
- **共通理解**: 型定義による設定仕様の明確化
- **レビュー効率**: 型安全性による品質保証
- **学習支援**: 新メンバーの設定理解促進

## 🔄 アップデート予定

- [ ] MMPose新バージョン対応
- [ ] 3Dポーズ推定設定の拡充
- [ ] カスタムデータセット型の追加
- [ ] 設定バリデーション機能
- [ ] 設定テンプレート生成機能

## 📚 関連ドキュメント

- [MMPose API Documentation](../docs/mmpose_api_documentation.md)
- [Single Person Optimization Guide](../docs/mmpose_single_person_optimization.md)
- [Configuration System Guide](../docs/mmpose_configuration_guide.md)

---

**注意**: このTypeDict定義は開発支援ツールです。実際のMMPose実行時には標準の辞書形式設定が使用されます。型定義により開発時の生産性と安全性が向上しますが、実行時動作は変わりません。