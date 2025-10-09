# VS Code Git自動同期設定

## 概要
このドキュメントは、VS Codeでプロジェクトを開いた際に現在のブランチを同名のリモートブランチと自動同期するための設定変更をまとめたものです。

## 変更日
2025年10月8日

## 設定ファイルの変更

### 1. `.vscode/settings.json` への追加設定

```json
{
    "git.autofetch": true,
    "git.autofetchPeriod": 180,
    "git.pullBeforeCheckout": true,
    "git.enableSmartCommit": true,
    "git.confirmSync": false,
    "git.fetchOnPull": true,
    "git.rebaseWhenSync": false,
    "git.branchSortOrder": "committerdate",
    "git.showPushSuccessNotification": true,
    "workbench.startupEditor": "none"
}
```

#### 設定項目の説明

| 設定項目 | 値 | 説明 |
|---------|---|------|
| `git.autofetch` | `true` | リモートの変更を自動的にフェッチ |
| `git.autofetchPeriod` | `180` | 自動フェッチの間隔（秒）- 3分間隔 |
| `git.pullBeforeCheckout` | `true` | ブランチ切り替え前に自動でプル |
| `git.enableSmartCommit` | `true` | スマートコミット機能を有効化 |
| `git.confirmSync` | `false` | 同期時の確認ダイアログを無効化 |
| `git.fetchOnPull` | `true` | プル実行時に自動でフェッチも実行 |
| `git.rebaseWhenSync` | `false` | 同期時はマージを使用（リベースしない） |
| `git.branchSortOrder` | `"committerdate"` | ブランチを最新コミット日順でソート |
| `git.showPushSuccessNotification` | `true` | プッシュ成功時に通知を表示 |
| `workbench.startupEditor` | `"none"` | 起動時にエディタを開かない |

### 2. `.vscode/tasks.json` の新規作成

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Git: Sync with remote",
            "type": "shell",
            "command": "pwsh.exe",
            "args": [
                "-Command",
                "$branch = git branch --show-current; $remote = @(git remote)[0]; if ($remote) { git pull $remote $branch; Write-Host 'Synced $branch with $remote' } else { Write-Host 'No remote found' -ForegroundColor Red }"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": []
        },
        {
            "label": "Git: Push to remote",
            "type": "shell",
            "command": "pwsh.exe",
            "args": [
                "-Command",
                "$branch = git branch --show-current; $remote = @(git remote)[0]; if ($remote) { git push $remote $branch; Write-Host 'Pushed $branch to $remote' } else { Write-Host 'No remote found' -ForegroundColor Red }"
            ],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": []
        }
    ]
}
```

#### タスクの説明

- **Git: Sync with remote**: 現在のブランチをリモートブランチから同期（プル）- リモート名を自動検出
- **Git: Push to remote**: 現在のブランチをリモートブランチにプッシュ - リモート名を自動検出
- **Git: Work branch auto commit and push**: work/*ブランチでの自動コミット＆プッシュ

#### 🔧 リモート名の自動検出機能

全てのタスクで`origin`ではなく、実際に設定されているリモート名を自動検出します：
- 現在のプロジェクトでは`plumiume`リモートが自動検出される
- 複数のリモートがある場合は最初のリモートを使用
- リモートが見つからない場合はエラーメッセージを表示

### 3. `.vscode/keybindings.json` の新規作成

```json
[
    {
        "key": "ctrl+shift+g ctrl+shift+s",
        "command": "workbench.action.tasks.runTask",
        "args": "Git: Sync with remote"
    },
    {
        "key": "ctrl+shift+g ctrl+shift+p",
        "command": "workbench.action.tasks.runTask",
        "args": "Git: Push to remote"
    }
]
```

#### キーバインドの説明

| キーバインド | 動作 |
|-------------|------|
| `Ctrl+Shift+G, Ctrl+Shift+S` | リモートブランチとの同期（プル） |
| `Ctrl+Shift+G, Ctrl+Shift+P` | リモートブランチへのプッシュ |

## この設定による効果

### 自動機能
1. **自動フェッチ**: プロジェクトを開くと3分間隔でリモートの変更情報を自動取得
2. **ブランチ切り替え時の自動プル**: ブランチを切り替える際に自動的にリモートの最新状態を取得
3. **スマートコミット**: ステージされていないファイルも含めて簡単にコミット可能

### 手動操作
1. **キーボードショートカット**: 設定したキーバインドで手動同期・プッシュが可能
2. **タスクランナー**: VS Codeのタスクメニューから同期タスクを実行可能

### UI改善
1. **通知**: プッシュ成功時に通知が表示される
2. **ブランチソート**: ブランチリストが最新のコミット日順で表示される
3. **確認ダイアログの簡素化**: 同期時の不要な確認ダイアログが無効化される

## 使用方法

### 基本的な使用法
1. VS Codeでプロジェクトを開く
2. 自動的にリモートの変更がフェッチされる
3. ブランチを切り替える際に自動的に最新状態に同期される

### 手動操作
- **同期**: `Ctrl+Shift+G, Ctrl+Shift+S` または コマンドパレット → "Tasks: Run Task" → "Git: Sync with remote"
- **プッシュ**: `Ctrl+Shift+G, Ctrl+Shift+P` または コマンドパレット → "Tasks: Run Task" → "Git: Push to remote"

## 注意事項

1. **リモートブランチの存在**: 同期を行うには、同名のリモートブランチが存在している必要があります
2. **コンフリクト**: リモートとローカルで変更が競合する場合は手動での解決が必要です
3. **ネットワーク**: 自動フェッチはネットワーク接続が必要です
4. **権限**: リモートリポジトリへの適切なアクセス権限が必要です

## Work/* ブランチでの自動コミット＆プッシュ機能

### 4. `.vscode/auto-commit.ps1` の新規作成

work/*ブランチでVS Codeを閉じる時に自動コミット＆プッシュを行うPowerShellスクリプト。

**主な機能:**
- 現在のブランチがwork/*パターンかチェック
- 変更があれば自動でコミット（WIP: Auto-save タイムスタンプ形式）
- リモートに自動プッシュ
- 変更がない場合はスキップ

### 5. `.vscode/auto-commit.bat` の新規作成

PowerShellスクリプトを簡単に実行するためのバッチファイル。

### 追加されたタスク

```json
{
    "label": "Git: Work branch auto commit and push",
    "type": "shell",
    "command": "pwsh.exe",
    "args": [
        "-Command",
        "$branch = git branch --show-current; if ($branch -like 'work/*') { $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'; git add -A; if (git diff --cached --quiet) { Write-Host 'No changes to commit' } else { git commit -m \"WIP: Auto-save $timestamp\"; git push origin $branch; Write-Host \"Auto-committed and pushed to $branch\" } } else { Write-Host \"Not on work/* branch (current: $branch)\" }"
    ]
}
```

### 追加されたキーバインド

| キーバインド | 動作 |
|-------------|------|
| `Ctrl+Shift+G, Ctrl+Shift+W` | Work/*ブランチの自動コミット＆プッシュ |
| `Ctrl+Alt+W` | Work/*ブランチの自動コミット＆プッシュ |

### 追加された設定項目

| 設定項目 | 値 | 説明 |
|---------|---|------|
| `files.saveAfterDelay` | `5000` | 5秒後に自動保存 |
| `files.autoSave` | `"afterDelay"` | 遅延後の自動保存を有効化 |
| `git.autoStash` | `true` | 自動スタッシュ機能を有効化 |
| `git.postCommitCommand` | `"push"` | コミット後に自動プッシュ |

## 使用方法（Work/*ブランチ機能）

### 自動実行
1. work/*ブランチで作業
2. ファイルが自動保存される（5秒後）
3. VS Codeを閉じる前に `Ctrl+Alt+W` で手動実行
4. または `Ctrl+Shift+G, Ctrl+Shift+W` で実行

### 手動実行
- **PowerShellから**: `.vscode\auto-commit.ps1 -Action commit`
- **バッチファイルから**: `.vscode\auto-commit.bat commit`
- **ステータス確認**: `.vscode\auto-commit.bat status`

### コミットメッセージ形式
```
WIP: Auto-save 2025-10-08 14:30:15
```

## トラブルシューティング

### よくある問題
- **認証エラー**: Gitの認証情報を確認してください
- **ネットワークエラー**: インターネット接続とリモートリポジトリのアクセス可能性を確認してください
- **コンフリクト**: マージコンフリクトが発生した場合は手動で解決してください
- **PowerShell実行ポリシー**: `Set-ExecutionPolicy RemoteSigned` で実行ポリシーを設定してください

### Work/*ブランチ機能のトラブルシューティング
- **スクリプトが実行されない**: PowerShellの実行ポリシーを確認
- **プッシュが失敗する**: リモートブランチの存在とアクセス権限を確認
- **ブランチが認識されない**: ブランチ名がwork/*パターンに合致するか確認

### 設定の無効化
設定を無効にしたい場合は、`settings.json`から該当する設定項目を削除するか、値を`false`に変更してください。

## Python仮想環境の自動アクティベーション設定

### 追加された設定項目

| 設定項目 | 値 | 説明 |
|---------|---|------|
| `python.terminal.activateEnvironment` | `true` | ターミナルで仮想環境を自動アクティベート |
| `python.terminal.activateEnvInCurrentTerminal` | `true` | 現在のターミナルでもアクティベート |
| `terminal.integrated.defaultProfile.windows` | `"PowerShell"` | デフォルトターミナルプロファイル |
| `terminal.integrated.profiles.windows` | カスタム設定 | 仮想環境を自動でアクティベートするPowerShellプロファイル |

### 追加されたタスク

- **Python: Setup Virtual Environment** - 仮想環境を作成（uvを使用）
- **Python: Show Virtual Environment Info** - 仮想環境情報を表示
- **Python: Activate Virtual Environment** - 新しいターミナルで仮想環境をアクティベート

### 追加されたキーバインド

| キーバインド | 動作 |
|-------------|------|
| `Ctrl+Shift+P, Ctrl+Shift+A` | 仮想環境をアクティベート |

### 仮想環境の場所

- **パス**: `D:/Users/ikeko/PythonVenvs/LMPipe/cp312`
- **理由**: OneDriveの同期を避けるため
- **構造**: プロジェクト名/Pythonバージョンで整理

### 手動アクティベーション方法

```powershell
# PowerShellで直接アクティベート
& 'D:/Users/ikeko/PythonVenvs/LMPipe/cp312/Scripts/Activate.ps1'

# 新しいPowerShellウィンドウでアクティベート
pwsh -NoExit -Command "& 'D:/Users/ikeko/PythonVenvs/LMPipe/cp312/Scripts/Activate.ps1'"

# スクリプト経由でアクティベーション情報表示
.vscode\setup-venv.ps1 -Action activate
```