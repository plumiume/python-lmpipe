## lmpipe — AI agent instructions (短く・具体的)

目的: リポジトリの主要構造、開発ワークフロー、プロジェクト固有のパターンを簡潔に示し、AIエージェントがすぐに貢献できるようにする。

### 1) ビッグピクチャ（主要コンポーネント）
- `src/lmpipe/estimator/` — 推論（Estimator）実装群。各推論器は `Estimator` を継承する。
- `src/lmpipe/interface/` — パイプラインのオーケストレーション（`LMPipeInterface`）。入力検出、executor 管理、collector 呼び出し、shutdown 処理を担う。
- `src/lmpipe/collector/` — 出力ハンドラ（landmarks, annotated frames 等）。`BaseCollector` を継承して実装。
- `src/lmpipe/plugins/loader.py` — プラグイン読み込み。`pyproject.toml` の `entry-points."lmpipe.plugins"` を参照して動的登録する。
- `src/lmpipe/app/cli/__main__.py` — CLI エントリポイント（console script `lmpipe`）。ここで `GlobalArgs.parse_args()`→estimator/plugin 選択→`LMPipeInterface` を起動する。

### 2) なぜこの構成か（短い理由）
- プラグイン化（entry points）で新しい推論器を外部パッケージとして追加可能にしている。
- `LMPipeInterface` はプロセス/スレッド分散（loky）と thread-local レジストリを用いて、ワーカー内で同じインターフェース・インスタンス参照を再構築する設計。
- Collector 層を抽象化し、形式（`.npy`, `.csv`, `cv2` 等）に応じて切り替えることで入出力の分離を実現。

### 3) 重要なワークフロー（開発者向けコマンド）
- インストール（開発モード）: `pip install -e .`（pyproject.toml の `project.scripts` により `lmpipe` コマンドが作成される）
- 実行（簡単）: `lmpipe` または `python -m lmpipe.app.cli`（CLI が `main()` を呼ぶ）
- ビルド: 本プロジェクトは `hatchling`（pyproject.toml の `build-backend`）を使います。通常の開発は `pip install -e .` で十分。

### 4) プロジェクト固有のパターンと注意点
- options: `src/lmpipe/options.py` は `clipar` の `@group` / `mixin.ReprMixin` を使う。CLI 引数は TypedDict/Group 構造で定義されるので修正時は型と group 定義を揃えてください。
- executor: `src/lmpipe/interface/__init__.py` は `DummyExecutor`（逐次実行）/`ProcessPoolExecutor`（loky ベース）を使う。`executor_mode` は `'batch'` または `'sample'`（`options['executor_mode']` を参照）で動作が切り替わる。
- collectors: `BaseCollector.apply_mode` の `ModeLiteral`（'skip'|'overwrite'|'postfix'）を守る。`LMPipeInterface._get_*` 系でフォーマット分岐（例: landmarks .npy/.csv/.json, annotated_frames_show_format 'cv2'）がある。
- プラグイン登録: `pyproject.toml` の `entry-points."lmpipe.plugins"` に `{type}.{name} = "module:ref"` 形式で登録する必要がある。`plugins.loader.load_plugins()` は entry point 名の末尾が type literal（例: `pose`, `left_hand` 等）であることを期待する。
- スレッド/プロセス間参照: `LMPipeInterface` は `_local.wv_pipelines`（WeakValueDictionary）を使い worker 初期化時にインスタンスを再登録する。ワーカー側でインターフェースを参照するラッパー（`_with_thread_local` / `_ThreadLocalMethod`）を使うこと。
- shutdown: `@shutdown_listener` デコレータが付いたメソッドは自動でシグナル時に呼ばれる。`_default_shutdown_listener` は executors の shutdown を行う。

### 5) よくある変更パターン（例とヒント）
- 新しい出力形式を追加する: `collector/*/writers.py` に実装を追加し、`LMPipeInterface._get_...` の `match` にケースを足す。例: `landmarks_matrix_save_format='.parquet'` を追加する場合は `writers` 側に `ParquetLandmarksMatrixWriter` を実装。
- 新しい推論器（plugin）を追加する: 新しいパッケージに entry point を追加（`pyproject.toml` の `entry-points."lmpipe.plugins"` に登録）。`loader.load_plugins()` が `(args, factory)` を受け取り、`factory(namespace)` が `Estimator` を返すことを期待する。
- CLI オプションを追加する: `src/lmpipe/options.py` の Group にフィールドを追加し、`app/cli/args.py`（CLI のパース定義）に反映する。

### 6) 外部依存と統合ポイント
- 主要依存: `mediapipe`, `opencv-python`, `loky`, `clipar`, `rich`（pyproject.toml を参照）
- plugin: `pyproject.toml` の `[project.entry-points."lmpipe.plugins"]` を通して外部パッケージが estimator を提供する

### 7) 短いコントラクト（AI 向けタスク仕様）
- 入力: 小さな変更（例: 新しい collector クラス、options フィールド、または plugin の loader 対応）
- 出力: 既存テストと型に沿う形での実装、`LMPipeInterface` の既存 hook（`configure_*`, initializers）を利用
- 失敗モード: CLI 実行で例外が出る場合はまず `executor_mode` と `max_workers`、および `entry-points` の不一致を確認

---
フィードバックください: 明確でない点や深掘りしてほしい箇所（特定のファイルや実装例）を教えてください。追加で `AGENT.md` スタイルの詳細ガイドも作成できます。
