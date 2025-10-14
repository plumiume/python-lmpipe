# MMPose Python API ドキュメント
フレームからランドマーク抽出のための包括的ガイド（可視化なし）

## 概要

MMPoseは、2D/3Dの人物・手・顔ランドマーク検出を可能にする強力なPythonライブラリです。このドキュメントでは、可視化機能を使用せずにフレーム画像からランドマークデータを抽出する方法について詳しく説明します。

## 対応する検出タイプ

| タイプ | 2D検出 | 3D検出 | 説明 |
|--------|--------|--------|------|
| 人物ポーズ | ✅ | ✅ | 体の主要関節点（17点など） |
| 手 | ✅ | ✅ | 手の関節点（21点など） |
| 顔 | ✅ | ❌ | 顔のランドマーク（68点など） |
| 全身 | ✅ | ❌ | 体・手・顔すべて含む |

## 1. 基本的なセットアップ

### 必要なライブラリのインポート

```python
import numpy as np
from mmpose.apis import MMPoseInferencer
import cv2
from typing import List, Dict, Any, Optional
```

### 基本的な設定

```python
# デバイス設定（GPU推奨）
device = 'cuda:0'  # CPUの場合は 'cpu'

# 各種閾値設定
bbox_thr = 0.3      # バウンディングボックス検出閾値
kpt_thr = 0.3       # キーポイント信頼度閾値
nms_thr = 0.3       # Non-Maximum Suppression閾値
```

## 2. MMPoseInferencerの初期化

### 2D人物ポーズ検出

```python
class HumanPose2DEstimator:
    def __init__(self, device='cuda:0'):
        self.inferencer = MMPoseInferencer(
            pose2d='human',  # 人物ポーズモデルエイリアス
            device=device
        )
    
    def estimate(self, image: np.ndarray) -> List[Dict]:
        """
        画像から2D人物ポーズを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            推定結果のリスト
        """
        # RGB形式に変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 推論実行
        result_generator = self.inferencer(
            image_rgb,
            show=False,
            return_vis=False
        )
        result = next(result_generator)
        
        return result['predictions'][0]
```

### 3D人物ポーズ検出

```python
class HumanPose3DEstimator:
    def __init__(self, device='cuda:0'):
        self.inferencer = MMPoseInferencer(
            pose3d='human3d',  # 3D人物ポーズモデルエイリアス
            device=device
        )
    
    def estimate(self, image: np.ndarray) -> List[Dict]:
        """
        画像から3D人物ポーズを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            推定結果のリスト（3D座標含む）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result_generator = self.inferencer(
            image_rgb,
            show=False,
            return_vis=False
        )
        result = next(result_generator)
        
        return result['predictions'][0]
```

### 2D手ランドマーク検出

```python
class HandPose2DEstimator:
    def __init__(self, device='cuda:0'):
        self.inferencer = MMPoseInferencer(
            pose2d='hand',  # 手ポーズモデルエイリアス
            device=device
        )
    
    def estimate(self, image: np.ndarray) -> List[Dict]:
        """
        画像から2D手ランドマークを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            推定結果のリスト（各手につき21点）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result_generator = self.inferencer(
            image_rgb,
            show=False,
            return_vis=False,
            bbox_thr=0.5,   # 手検出の閾値を調整
            kpt_thr=0.05    # キーポイント閾値を調整
        )
        result = next(result_generator)
        
        return result['predictions'][0]
```

### 3D手ランドマーク検出

```python
class HandPose3DEstimator:
    def __init__(self, device='cuda:0'):
        self.inferencer = MMPoseInferencer(
            pose3d='hand3d',  # 3D手ポーズモデルエイリアス
            device=device
        )
    
    def estimate(self, image: np.ndarray) -> List[Dict]:
        """
        画像から3D手ランドマークを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            推定結果のリスト（3D座標含む）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result_generator = self.inferencer(
            image_rgb,
            show=False,
            return_vis=False
        )
        result = next(result_generator)
        
        return result['predictions'][0]
```

### 2D顔ランドマーク検出

```python
class FacePose2DEstimator:
    def __init__(self, device='cuda:0'):
        self.inferencer = MMPoseInferencer(
            pose2d='face',  # 顔ランドマークモデルエイリアス
            device=device
        )
    
    def estimate(self, image: np.ndarray) -> List[Dict]:
        """
        画像から2D顔ランドマークを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            推定結果のリスト（顔につき68点など）
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result_generator = self.inferencer(
            image_rgb,
            show=False,
            return_vis=False,
            radius=1        # 顔の小さなランドマーク用
        )
        result = next(result_generator)
        
        return result['predictions'][0]
```

## 3. 統合されたランドマーク推定器

```python
class UnifiedLandmarkEstimator:
    """
    人物・手・顔のランドマークを統合して推定するクラス
    """
    
    def __init__(self, device='cuda:0'):
        self.device = device
        self.human_estimator = HumanPose2DEstimator(device)
        self.hand_estimator = HandPose2DEstimator(device) 
        self.face_estimator = FacePose2DEstimator(device)
    
    def estimate_all(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像からすべてのランドマークを推定
        
        Args:
            image: 入力画像 (H, W, 3) BGR形式
            
        Returns:
            すべてのランドマーク推定結果
        """
        results = {
            'human_pose': [],
            'hand_landmarks': [],
            'face_landmarks': []
        }
        
        try:
            # 人物ポーズ推定
            human_results = self.human_estimator.estimate(image)
            results['human_pose'] = human_results
        except Exception as e:
            print(f"Human pose estimation failed: {e}")
        
        try:
            # 手ランドマーク推定
            hand_results = self.hand_estimator.estimate(image)
            results['hand_landmarks'] = hand_results
        except Exception as e:
            print(f"Hand landmark estimation failed: {e}")
        
        try:
            # 顔ランドマーク推定
            face_results = self.face_estimator.estimate(image)
            results['face_landmarks'] = face_results
        except Exception as e:
            print(f"Face landmark estimation failed: {e}")
        
        return results
```

## 4. 実用的な使用例

### 単一画像での使用例

```python
def process_single_image(image_path: str):
    """
    単一画像を処理してランドマークを抽出
    """
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # 推定器初期化
    estimator = UnifiedLandmarkEstimator(device='cuda:0')
    
    # ランドマーク推定
    results = estimator.estimate_all(image)
    
    # 結果の処理
    print(f"Detected {len(results['human_pose'])} human poses")
    print(f"Detected {len(results['hand_landmarks'])} hands")
    print(f"Detected {len(results['face_landmarks'])} faces")
    
    return results

# 使用例
results = process_single_image('path/to/your/image.jpg')
```

### 動画フレーム処理

```python
def process_video_frames(video_path: str, output_file: str = None):
    """
    動画の各フレームを処理してランドマークを抽出
    """
    cap = cv2.VideoCapture(video_path)
    estimator = UnifiedLandmarkEstimator(device='cuda:0')
    
    frame_results = []
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレーム処理
            results = estimator.estimate_all(frame)
            
            # フレーム番号と結果を保存
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS),
                'landmarks': results
            }
            frame_results.append(frame_data)
            
            frame_count += 1
            
            # 進捗表示
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
    
    finally:
        cap.release()
    
    # 結果をファイルに保存（オプション）
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(frame_results, f, indent=2, default=str)
    
    return frame_results

# 使用例
results = process_video_frames('path/to/your/video.mp4', 'landmarks_output.json')
```

## 5. ランドマークデータの抽出と処理

### ランドマーク座標の抽出

```python
def extract_keypoints(predictions: List[Dict]) -> np.ndarray:
    """
    推定結果からキーポイント座標を抽出
    
    Args:
        predictions: MMPose推定結果
        
    Returns:
        キーポイント座標配列 (N, num_keypoints, 2 or 3)
    """
    if not predictions:
        return np.array([])
    
    keypoints_list = []
    for pred in predictions:
        if 'keypoints' in pred:
            keypoints_list.append(pred['keypoints'])
    
    if keypoints_list:
        return np.array(keypoints_list)
    else:
        return np.array([])

def extract_keypoint_scores(predictions: List[Dict]) -> np.ndarray:
    """
    推定結果からキーポイント信頼度を抽出
    
    Args:
        predictions: MMPose推定結果
        
    Returns:
        キーポイント信頼度配列 (N, num_keypoints)
    """
    if not predictions:
        return np.array([])
    
    scores_list = []
    for pred in predictions:
        if 'keypoint_scores' in pred:
            scores_list.append(pred['keypoint_scores'])
    
    if scores_list:
        return np.array(scores_list)
    else:
        return np.array([])
```

### 品質フィルタリング

```python
def filter_high_quality_landmarks(
    keypoints: np.ndarray, 
    scores: np.ndarray, 
    score_threshold: float = 0.5
) -> tuple:
    """
    信頼度に基づいてランドマークをフィルタリング
    
    Args:
        keypoints: キーポイント座標
        scores: キーポイント信頼度  
        score_threshold: 信頼度閾値
        
    Returns:
        フィルタリング後の (keypoints, scores)
    """
    if len(keypoints) == 0 or len(scores) == 0:
        return keypoints, scores
    
    # 平均信頼度を計算
    mean_scores = np.mean(scores, axis=1)
    
    # 閾値以上の結果のみ保持
    valid_mask = mean_scores >= score_threshold
    
    filtered_keypoints = keypoints[valid_mask]
    filtered_scores = scores[valid_mask]
    
    return filtered_keypoints, filtered_scores
```

## 6. パフォーマンス最適化

### バッチ処理

```python
class BatchLandmarkEstimator:
    """
    複数画像を効率的にバッチ処理するクラス
    """
    
    def __init__(self, pose_type='human', device='cuda:0', batch_size=8):
        self.inferencer = MMPoseInferencer(
            pose2d=pose_type,
            device=device
        )
        self.batch_size = batch_size
    
    def estimate_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        複数画像をバッチ処理で推定
        
        Args:
            images: 画像リスト
            
        Returns:
            推定結果リスト
        """
        all_results = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # RGB変換
            batch_rgb = [
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                for img in batch_images
            ]
            
            # バッチ推論
            for img in batch_rgb:
                result_generator = self.inferencer(
                    img,
                    show=False,
                    return_vis=False
                )
                result = next(result_generator)
                all_results.append(result['predictions'][0])
        
        return all_results
```

### メモリ効率化

```python
def process_large_video_efficient(
    video_path: str,
    output_callback,
    frame_skip: int = 1,
    max_frames: Optional[int] = None
):
    """
    大容量動画を効率的に処理（メモリ使用量を抑制）
    
    Args:
        video_path: 動画ファイルパス
        output_callback: 結果処理用コールバック関数
        frame_skip: フレームスキップ間隔
        max_frames: 処理する最大フレーム数
    """
    cap = cv2.VideoCapture(video_path)
    estimator = UnifiedLandmarkEstimator(device='cuda:0')
    
    frame_count = 0
    processed_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレームスキップ
            if frame_count % (frame_skip + 1) != 0:
                frame_count += 1
                continue
            
            # 推定実行
            results = estimator.estimate_all(frame)
            
            # 結果をコールバックで処理（メモリに蓄積しない）
            frame_data = {
                'frame_number': frame_count,
                'results': results
            }
            output_callback(frame_data)
            
            processed_count += 1
            frame_count += 1
            
            # 最大フレーム数チェック
            if max_frames and processed_count >= max_frames:
                break
    
    finally:
        cap.release()

# 使用例：結果をファイルにストリーミング保存
def save_result_callback(frame_data):
    """結果保存用コールバック"""
    with open('streaming_results.jsonl', 'a') as f:
        import json
        f.write(json.dumps(frame_data, default=str) + '\n')

process_large_video_efficient(
    'large_video.mp4',
    save_result_callback,
    frame_skip=2  # 2フレームごとに処理
)
```

## 7. エラーハンドリング

### 堅牢なエラー処理

```python
class RobustLandmarkEstimator:
    """
    エラーハンドリングを含む堅牢なランドマーク推定器
    """
    
    def __init__(self, device='cuda:0', retry_attempts=3):
        self.device = device
        self.retry_attempts = retry_attempts
        self.estimators = {}
        
    def _get_estimator(self, pose_type: str):
        """推定器を遅延初期化"""
        if pose_type not in self.estimators:
            try:
                self.estimators[pose_type] = MMPoseInferencer(
                    pose2d=pose_type,
                    device=self.device
                )
            except Exception as e:
                print(f"Failed to initialize {pose_type} estimator: {e}")
                return None
        
        return self.estimators[pose_type]
    
    def estimate_with_retry(
        self, 
        image: np.ndarray, 
        pose_type: str = 'human'
    ) -> Optional[List[Dict]]:
        """
        リトライ機能付き推定
        """
        estimator = self._get_estimator(pose_type)
        if estimator is None:
            return None
        
        for attempt in range(self.retry_attempts):
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                result_generator = estimator(
                    image_rgb,
                    show=False,
                    return_vis=False
                )
                result = next(result_generator)
                
                return result['predictions'][0]
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    print(f"All attempts failed for {pose_type}")
                    return None
                
                # 短時間待機してリトライ
                import time
                time.sleep(0.1)
        
        return None
```

## 8. 設定とカスタマイズ

### カスタムモデル設定

```python
def create_custom_inferencer(
    config_path: str,
    checkpoint_path: str,
    device: str = 'cuda:0'
):
    """
    カスタムモデルを使用した推定器を作成
    
    Args:
        config_path: モデル設定ファイルパス
        checkpoint_path: モデル重みファイルパス
        device: 使用デバイス
        
    Returns:
        カスタム推定器
    """
    inferencer = MMPoseInferencer(
        pose2d=config_path,
        pose2d_weights=checkpoint_path,
        device=device
    )
    
    return inferencer

# 使用例
custom_estimator = create_custom_inferencer(
    config_path='path/to/custom_config.py',
    checkpoint_path='path/to/custom_weights.pth'
)
```

### 検出パラメータの調整

```python
def create_tuned_inferencer(pose_type: str, **kwargs):
    """
    検出パラメータを調整した推定器を作成
    
    Args:
        pose_type: ポーズタイプ ('human', 'hand', 'face')
        **kwargs: その他のパラメータ
        
    Returns:
        調整済み推定器
    """
    default_params = {
        'human': {
            'bbox_thr': 0.3,
            'kpt_thr': 0.3,
        },
        'hand': {
            'bbox_thr': 0.5,
            'kpt_thr': 0.05,
        },
        'face': {
            'bbox_thr': 0.6,
            'kpt_thr': 0.1,
        }
    }
    
    params = default_params.get(pose_type, {})
    params.update(kwargs)
    
    return MMPoseInferencer(pose2d=pose_type, **params)
```

## 9. 使用可能なモデルエイリアス

### 2Dモデル

| エイリアス | 説明 | ポーズ推定器 | 検出器 |
|------------|------|--------------|--------|
| human | 人物ポーズ推定 | RTMPose-m | RTMDet-m |
| hand | 手ランドマーク検出 | RTMPose-m | ssdlite_mobilenetv2 |
| face | 顔ランドマーク検出 | RTMPose-m | yolox-s |
| wholebody | 全身ポーズ推定 | RTMPose-m | RTMDet-m |
| animal | 動物ポーズ推定 | RTMPose-m | RTMDet-m |

### 3Dモデル

| エイリアス | 説明 | ポーズ推定器 | 検出器 |
|------------|------|--------------|--------|
| human3d | 3D人物ポーズ推定 | MotionBert | RTMPose-m + RTMDet-m |
| hand3d | 3D手ポーズ推定 | InterNet | 全画像 |

## 10. トラブルシューティング

### よくある問題と解決方法

#### GPU メモリ不足
```python
# CPUフォールバック
try:
    estimator = UnifiedLandmarkEstimator(device='cuda:0')
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("GPU memory insufficient, falling back to CPU")
        estimator = UnifiedLandmarkEstimator(device='cpu')
    else:
        raise e
```

#### モデル読み込み失敗
```python
# ネットワーク接続チェック
import urllib.request

def check_model_availability(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=10)
        return True
    except:
        return False

# ローカルモデル使用
if not check_model_availability("https://download.openmmlab.com/..."):
    print("Using local model files")
    # ローカルファイルパスを指定
```

## 11. MMPoseInferencer内部アーキテクチャ詳解

### 11.1 全体構造

MMPoseInferencerは統合インターフェースとして機能し、内部的には専用の推定器クラスに処理を委譲する階層構造になっています。

```
MMPoseInferencer (統合インターフェース)
├── Pose2DInferencer (2D姿勢推定)
│   ├── TopdownPoseEstimator (ポーズ推定モデル)
│   └── DetInferencer (物体検出器)
├── Pose3DInferencer (3D姿勢推定)
│   ├── MotionBert/VideoPose3D (3Dモデル)
│   ├── Pose2DInferencer (2D前処理)
│   └── DetInferencer (物体検出器)
└── Hand3DInferencer (3D手推定)
    ├── InterNet (3D手モデル)
    └── 全画像処理（検出器なし）
```

### 11.2 内部推定器の選択ロジック

```python
def analyze_inferencer_selection():
    """
    MMPoseInferencerがどの内部推定器を選択するかを解析
    """
    selection_logic = {
        # 3D優先の選択ロジック
        'pose3d_provided': {
            'hand3d': 'Hand3DInferencer',
            'other_3d': 'Pose3DInferencer'
        },
        'pose2d_only': 'Pose2DInferencer',
        'neither': 'ValueError'
    }
    
    # 実際の選択例
    examples = {
        "MMPoseInferencer('human')": "Pose2DInferencer",
        "MMPoseInferencer('hand')": "Pose2DInferencer", 
        "MMPoseInferencer(pose3d='human3d')": "Pose3DInferencer",
        "MMPoseInferencer(pose3d='hand3d')": "Hand3DInferencer"
    }
    
    return selection_logic, examples
```

### 11.3 モデルエイリアスシステム

#### 利用可能なモデルエイリアス（2025年10月現在）

```python
# 実際のMMPoseエイリアス一覧
MODEL_ALIASES = {
    # 2D人物ポーズ
    'human': 'rtmpose-m_8xb256-420e_body8-256x192',
    'body': 'rtmpose-m_8xb256-420e_body8-256x192', 
    'body17': 'rtmpose-m_8xb256-420e_body8-256x192',
    'body26': 'rtmpose-m_8xb512-700e_body8-halpe26-256x192',
    'wholebody': 'rtmw-m_8xb1024-270e_cocktail14-256x192',
    
    # 2D手ランドマーク
    'hand': 'rtmpose-m_8xb256-210e_hand5-256x256',
    
    # 2D顔ランドマーク  
    'face': 'rtmpose-m_8xb64-120e_lapa-256x256',
    
    # 動物ポーズ
    'animal': 'rtmpose-m_8xb64-210e_ap10k-256x256',
    
    # 3Dモデル
    'human3d': 'motionbert_dstformer-ft-243frm_8xb32-120e_h36m',
    'hand3d': 'internet_res50_4xb16-20e_interhand3d-256x256',
    
    # ViTPoseバリエーション
    'vitpose': 'td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192',
    'vitpose-s': 'td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192',
    'vitpose-b': 'td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192',
    'vitpose-l': 'td-hm_ViTPose-large-simple_8xb64-210e_coco-256x192',
    'vitpose-h': 'td-hm_ViTPose-huge-simple_8xb64-210e_coco-256x192',
    
    # その他
    'rtmo': 'rtmo-l_16xb16-600e_body7-640x640',
    'rtmpose-l': 'rtmpose-l_8xb256-420e_body8-384x288',
    'edpose': 'edpose_res50_8xb2-50e_coco-800x1333'
}
```

#### エイリアス解決システム

```python
class ModelAliasResolver:
    """
    モデルエイリアスの解決プロセスを実装
    """
    
    @staticmethod
    def resolve_alias(alias: str) -> dict:
        """
        エイリアスから実際のモデル設定を解決
        
        解決プロセス:
        1. メタファイルからモデル設定リストを取得
        2. エイリアスフィールドとマッチング
        3. 対応する設定ファイル名を返却
        """
        from mmengine.infer import BaseInferencer
        from mmpose.apis.inferencers.utils import get_model_aliases
        
        # メタファイルベースの解決
        aliases = get_model_aliases('mmpose')
        
        if alias in aliases:
            config_name = aliases[alias]
            return {
                'alias': alias,
                'config_name': config_name,
                'resolved': True
            }
        else:
            return {
                'alias': alias,
                'config_name': None,
                'resolved': False
            }
```

### 11.4 デフォルト検出器システム

#### 各ポーズタイプの専用検出器

```python
DEFAULT_DETECTORS = {
    'human/body/wholebody': {
        'model': 'RTMDet-M (Person Detection)',
        'config': 'rtmdet_m_640-8xb32_coco-person.py',
        'weights': 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        'category_ids': [0],  # 人物カテゴリ
        'description': '人物検出に最適化されたRTMDet中型モデル'
    },
    'hand': {
        'model': 'RTMDet-Nano (Hand Detection)',
        'config': 'rtmdet_nano_320-8xb32_hand.py', 
        'weights': 'rtmdet_nano_8xb32-300e_hand-267f9c8f.pth',
        'category_ids': [0],  # 手カテゴリ
        'description': '手検出専用の軽量RTMDetモデル'
    },
    'face': {
        'model': 'YOLOX-S (Face Detection)',
        'config': 'yolox-s_8xb8-300e_coco-face.py',
        'weights': 'yolo-x_8xb8-300e_coco-face_13274d7c.pth', 
        'category_ids': [0],  # 顔カテゴリ
        'description': '顔検出専用のYOLOXモデル'
    },
    'animal': {
        'model': 'RTMDet-M (Multi-Animal)',
        'config': 'rtmdet-m',
        'weights': None,  # デフォルトウェイト使用
        'category_ids': [15, 16, 17, 18, 19, 20, 21, 22, 23],  # 動物カテゴリ
        'description': '複数種類の動物検出対応'
    }
}
```

#### 検出器初期化プロセス

```python
class DetectorInitializer:
    """
    検出器の初期化プロセスを管理
    """
    
    def __init__(self, pose_type: str):
        self.pose_type = pose_type
        
    def initialize_detector(self, det_model=None, det_weights=None, det_cat_ids=None):
        """
        検出器初期化の詳細プロセス
        
        1. ポーズタイプからデフォルト検出器を決定
        2. カスタム検出器パラメータが指定されている場合はオーバーライド
        3. 'whole_image'指定時は検出器を無効化
        4. MMDetection DetInferencerとして初期化
        """
        # ステップ1: データセットタイプから推論
        dataset_type = self._infer_dataset_type()
        
        # ステップ2: デフォルト検出器設定取得
        if det_model is None:
            default_config = DEFAULT_DETECTORS.get(dataset_type)
            if default_config:
                det_model = default_config['config']
                det_weights = det_weights or default_config['weights']
                det_cat_ids = det_cat_ids or default_config['category_ids']
        
        # ステップ3: 検出器インスタンス作成
        if det_model not in ('whole_image', 'whole-image'):
            from mmdet.apis.det_inferencer import DetInferencer
            return DetInferencer(
                model=det_model,
                weights=det_weights, 
                device=self.device,
                scope='mmdet'
            )
        else:
            return None  # 全画像処理モード
```

### 11.5 処理フローの詳細

#### 2D推定処理フロー

```python
def detailed_2d_inference_flow():
    """
    2D推定の詳細な処理フロー
    """
    flow = {
        'step_1_initialization': {
            'action': 'Pose2DInferencer作成',
            'details': [
                'モデルエイリアス解決',
                'ポーズ推定モデル読み込み', 
                '検出器初期化（Top-downの場合）',
                'ビジュアライザー設定'
            ]
        },
        'step_2_preprocessing': {
            'action': '前処理実行',
            'details': [
                '入力形式検証（画像/動画/Webカメラ）',
                '物体検出実行（Top-downの場合）',
                'バウンディングボックス取得',
                'NMS（Non-Maximum Suppression）適用',
                '画像正規化・リサイズ'
            ]
        },
        'step_3_forward': {
            'action': 'モデル推論',
            'details': [
                '前処理済み画像をモデルに入力',
                'キーポイント座標と信頼度を推定',
                '結果をPoseDataSample形式に変換'
            ]
        },
        'step_4_postprocessing': {
            'action': '後処理',
            'details': [
                'キーポイント座標の逆変換',
                '信頼度に基づくフィルタリング',
                'NMS（キーポイントベース）適用',
                '結果マージ・整理'
            ]
        }
    }
    return flow
```

#### 3D推定処理フロー

```python
def detailed_3d_inference_flow():
    """
    3D推定の詳細な処理フロー
    """
    flow = {
        'step_1_2d_estimation': {
            'action': '2Dポーズ推定',
            'details': [
                '内部でPose2DInferencerを使用',
                '2Dキーポイントを取得'
            ]
        },
        'step_2_3d_lifting': {
            'action': '2D-to-3Dリフティング',
            'details': [
                '2Dキーポイントを正規化',
                '3Dポーズ推定モデル（MotionBert等）に入力',
                '3D座標とz軸深度を推定'
            ]
        },
        'step_3_post_processing': {
            'action': '3D後処理',
            'details': [
                'ルートジョイント正規化',
                'カメラパラメータ考慮',
                '3D座標の最終調整'
            ]
        }
    }
    return flow
```

### 11.6 内部コンポーネント詳解

#### ポーズ推定モデル

```python
class PoseEstimatorComponents:
    """
    ポーズ推定モデルの内部コンポーネント
    """
    
    RTMPose_ARCHITECTURE = {
        'backbone': 'CSPNeXt',
        'neck': 'CSPNeXtPAFPN',  
        'head': 'RTMCCHead',
        'loss': 'KLDiscretLoss',
        'codec': 'SimCCLabel',
        'description': '高速・高精度なリアルタイムポーズ推定器'
    }
    
    VITPOSE_ARCHITECTURE = {
        'backbone': 'Vision Transformer',
        'neck': 'FeaturePyramidNetwork',
        'head': 'TopdownHeatmapSimpleHead', 
        'loss': 'JointsMSELoss',
        'codec': 'MSRAHeatmap',
        'description': 'Vision Transformerベースの高精度モデル'
    }
```

#### データ処理パイプライン

```python
class DataProcessingPipeline:
    """
    データ処理パイプラインの詳細
    """
    
    PREPROCESSING_PIPELINE = [
        'LoadImage',           # 画像読み込み
        'DetectionInference',  # 物体検出
        'TopdownAffine',       # アフィン変換
        'GenerateTarget',      # ターゲット生成
        'PackPoseInputs'       # 入力パッキング
    ]
    
    POSTPROCESSING_PIPELINE = [
        'DecodeKeypoints',     # キーポイントデコード  
        'ApplyNMS',           # NMS適用
        'FilterByScore',      # 信頼度フィルタ
        'CoordinateTransform', # 座標変換
        'PackResults'         # 結果パッケージング
    ]
```

### 11.7 実践的な内部調査方法

#### 内部状態の確認

```python
def inspect_inferencer_internals(inferencer):
    """
    MMPoseInferencerの内部状態を調査
    """
    print("=== MMPoseInferencer内部構造調査 ===")
    
    # 基本情報
    print(f"推定器タイプ: {type(inferencer.inferencer)}")
    
    # モデル情報
    model = inferencer.inferencer.model
    print(f"モデルタイプ: {type(model)}")
    print(f"データモード: {inferencer.inferencer.cfg.data_mode}")
    
    # 検出器情報
    if hasattr(inferencer.inferencer, 'detector'):
        detector = inferencer.inferencer.detector
        if detector is not None:
            print(f"検出器タイプ: {type(detector)}")
            print(f"検出器モデル: {detector.model}")
        else:
            print("検出器: 使用しない（全画像モード）")
    
    # データセット情報
    if hasattr(model, 'dataset_meta'):
        meta = model.dataset_meta
        print(f"データセット: {meta.get('dataset_name', 'Unknown')}")
        print(f"キーポイント数: {len(meta.get('joint_weights', []))}")
        print(f"スケルトン接続: {len(meta.get('skeleton_info', []))}本")
    
    # パイプライン情報
    if hasattr(inferencer.inferencer, 'pipeline'):
        pipeline = inferencer.inferencer.pipeline
        print(f"前処理パイプライン: {[step.__class__.__name__ for step in pipeline.transforms]}")

# 使用例
inferencer = MMPoseInferencer('human')
inspect_inferencer_internals(inferencer)
```

#### 処理時間分析

```python
def analyze_processing_time(inferencer, image):
    """
    各処理段階の時間を分析
    """
    import time
    
    timings = {}
    
    # 前処理時間
    start_time = time.time()
    preprocessed = list(inferencer.preprocess([image]))
    timings['preprocessing'] = time.time() - start_time
    
    # 推論時間
    start_time = time.time()
    results = []
    for data in preprocessed:
        result = inferencer.forward(data)
        results.append(result)
    timings['inference'] = time.time() - start_time
    
    # 後処理時間
    start_time = time.time()
    final_results = inferencer.postprocess(results)
    timings['postprocessing'] = time.time() - start_time
    
    return timings

# パフォーマンス分析例
import cv2
image = cv2.imread('sample_image.jpg')
inferencer = MMPoseInferencer('human')
timings = analyze_processing_time(inferencer, image)
print("処理時間分析:", timings)
```

## まとめ

このドキュメントでは、MMPose Python APIを使用してフレームから各種ランドマーク（人物ポーズ、手、顔の2D/3D）を効率的に抽出する方法を説明しました。可視化機能を使用せずに純粋にデータ抽出に特化した実装例を提供しており、実際のアプリケーション開発に直接活用できます。

### 主要なポイント
- **統合API**: MMPoseInferencerで複数の検出タイプに対応
- **階層アーキテクチャ**: 内部的に専用推定器クラスを使用
- **モデルエイリアス**: メタファイルベースの柔軟なモデル選択
- **自動検出器選択**: ポーズタイプに応じた最適な検出器の自動設定
- **効率性**: バッチ処理とメモリ効率化で大規模データ処理
- **堅牢性**: エラーハンドリングとリトライ機構
- **柔軟性**: カスタムモデルと検出パラメータの調整

### MMPoseInferencer内部理解の重要性
1. **パフォーマンス最適化**: 各コンポーネントの特性理解による効率化
2. **カスタマイズ**: 内部構造の把握による高度なカスタマイズ
3. **デバッグ**: 問題発生時の原因特定と解決
4. **拡張性**: 新しい機能やモデルの統合

### 推奨される使用パターン
1. **研究・開発**: 単一画像での詳細分析
2. **バッチ処理**: 大量画像の一括処理
3. **リアルタイム**: 動画フレームのストリーミング処理
4. **プロダクション**: 堅牢なエラーハンドリング付き実装
5. **カスタム開発**: 内部アーキテクチャを活用した高度なカスタマイズ