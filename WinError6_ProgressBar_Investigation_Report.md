# WinError 6 (ハンドルが無効です) 調査レポート

**日付**: 2025年10月27日  
**対象ファイル**: `src/lmpipe/app/cli/progress_bar.py`  
**エラー**: `OSError: [WinError 6] ハンドルが無効です。`

---

## 問題の概要

バッチモード実行時に、`multiprocessing.connection.PipeConnection.__del__`で`WinError 6`が発生します。これは、同一のWindowsハンドルが複数回クローズされようとしたことを示しています。

```
Exception ignored in: <function _ConnectionBase.__del__ at 0x...>
Traceback (most recent call last):
  File "...\multiprocessing\connection.py", line 133, in __del__
    self._close()
  File "...\multiprocessing\connection.py", line 282, in _close
    _CloseHandle(self._handle)
OSError: [WinError 6] ハンドルが無効です。
```

---

## 根本原因の分析

### 1. **Queue オブジェクトの重複参照**

`ProgressClient`のシリアライゼーション処理で、`Queue`オブジェクトが複数の経路で子プロセスに渡されています:

**問題のあるフロー:**

```python
# ProgressClient.__getstate__ (行 430-452)
def __getstate__(self) -> dict[str, Any]:
    with self._manager._serialize_with_client(self) as spawning_context:
        pre_serialized_manager = pickle.dumps(self._manager)
    
    return {
        **self.__dict__,                    # ① self._map_q を含む
        '_manager': pre_serialized_manager, # ② ProgressManager内に _reduce_q, _map_qs
        'spawning_context': spawning_context, # ③ 同じ Queue オブジェクトへの参照
    }
```

**問題点:**
- `spawning_context.reduce_q` と `self._manager._reduce_q` が **同一オブジェクト**
- `spawning_context.map_qs[0]` と `self._map_q` が **同一オブジェクト**
- Pickle化により、同じQueueが複数回シリアライズ/デシリアライズされる

### 2. **PipeConnection ハンドルの重複**

`Queue`オブジェクトは内部で`PipeConnection`を使用しており、以下の構造を持ちます:

```
Queue
├── _reader: PipeConnection (Windows Handle: 0x1234)
└── _writer: PipeConnection (Windows Handle: 0x5678)
```

**デシリアライズ時の問題:**
1. 子プロセスで`ProgressClient`がデシリアライズされる
2. `spawning_context`内の`Queue`が復元される（PipeConnection含む）
3. `self._manager`が復元され、再び同じ`Queue`参照が作られる
4. `self._map_q`も復元される

結果として、**同一のPipeConnectionハンドルに対して複数のPythonオブジェクトが作成**されます。

### 3. **ガベージコレクション時のハンドルクローズ競合**

プロセス終了時やGC時に:
1. 最初の`PipeConnection.__del__`が呼ばれ、Windowsハンドルをクローズ
2. 2つ目の`PipeConnection.__del__`が呼ばれるが、ハンドルは既に無効
3. `_CloseHandle`が失敗し、`WinError 6`が発生

---

## 現在の設計の意図と問題

### `_ProgressManagerSpawningContext.override_state` の意図

```python
@classmethod
def override_state(cls, state: dict[str, Any]) -> dict[str, Any]:
    return {
        'reduce_q': _NeedsRestoreDescriptor('reduce_q', _is_queue),
        'map_qs': {},
    }
```

この関数は、`ProgressManager.__getstate__`で`_reduce_q`を`_NeedsRestoreDescriptor`に置き換えようとしていますが、**実際には使用されていません**。

**現在の実装 (行 357-374):**
```python
def __getstate__(self) -> dict[str, Any]:
    if not self._with_client:
        raise RuntimeError(...)

    state: dict[str, Any] = {
        **self.__dict__,
        'manager_thread': None,
        '_progress_registry': {},
    }

    if self._with_client:
        state['_is_serialized_without_mp_spawning'] = True
        _ProgressManagerSpawningContext.override_state(state)  # ← 返り値を使っていない！

    return state
```

**問題:** `override_state`の返り値が無視され、`state`が更新されていません。

---

## 推奨される修正方法

### **オプション A: override_state を正しく適用する**

```python
def __getstate__(self) -> dict[str, Any]:
    if not self._with_client:
        raise RuntimeError(...)

    state: dict[str, Any] = {
        **self.__dict__,
        'manager_thread': None,
        '_progress_registry': {},
    }

    if self._with_client:
        state['_is_serialized_without_mp_spawning'] = True
        state.update(_ProgressManagerSpawningContext.override_state(state))  # FIX

    return state
```

**効果:**
- `ProgressManager`内の`_reduce_q`と`_map_qs`がシリアライズされなくなる
- `spawning_context`経由でのみQueueが渡される
- Queue参照が一本化され、重複ハンドルが解消される

### **オプション B: spawning_context を別の方法で渡す**

`ProgressClient.__getstate__`で`spawning_context`を別途保存せず、`_manager`の復元後に再構築する方法も考えられますが、より複雑になります。

---

## 追加の推奨事項

### 1. **デバッグログの保持**

追加したデバッグログ（`__getstate__`, `__setstate__`, `_store_spawning_context`）を一時的に保持し、修正後の動作確認に使用してください。

### 2. **テストケースの作成**

- 単一ワーカーでのシリアライゼーションテスト
- 複数ワーカーでのQueue共有テスト
- プロセス終了時のクリーンアップテスト

### 3. **Windows固有の問題への対応**

この問題はWindowsのハンドル管理に起因しているため、Linuxでは発生しない可能性があります。クロスプラットフォームテストを実施してください。

---

## 修正の優先度

**高**: この問題はエラーメッセージを出力しますが、`Exception ignored in __del__`のため、プログラムの主要機能には影響しません。ただし、リソースリークやデバッグの妨げになるため、早急な修正を推奨します。

---

## 検証手順

修正後、以下のコマンドで問題が解消されることを確認してください:

```powershell
uv run lmpipe --executor-mode batch --max-workers 12 \
  --landmarks-matrix-save-format .npy \
  <INPUT_DIR> <OUTPUT_DIR> \
  holistic pose mediapipe --min-pose-detection-confidence 0.0 \
  left_hand mediapipe right_hand mediapipe
```

**期待される結果:**
- `Exception ignored in: <function _ConnectionBase.__del__>` が出力されない
- デバッグログで同一のQueue IDが確認される
- 正常に処理が完了する

---

## まとめ

**根本原因**: `ProgressManager.__getstate__`で`override_state`の返り値が適用されず、Queueオブジェクトが重複してシリアライズされることで、PipeConnectionハンドルの重複クローズが発生。

**推奨修正**: `state.update(_ProgressManagerSpawningContext.override_state(state))`を追加し、Queue参照を一本化する。
