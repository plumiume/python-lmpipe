# Todo Tree 設定項目（日本語）

Todo Tree拡張機能の主要設定項目を日本語で説明します。

## 一般設定 (General Settings)

### `todo-tree.general.debug`
- **日本語名**: デバッグモード
- **説明**: デバッグ情報を出力チャネルに表示します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.general.enableFileWatcher`
- **日本語名**: ファイル監視を有効化
- **説明**: ファイル変更時に自動的にツリーを更新します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.general.rootFolder`
- **日本語名**: ルートフォルダ
- **説明**: TODOを検索するルートフォルダを指定します
- **デフォルト値**: `""`
- **型**: `string`

### `todo-tree.general.statusBar`
- **日本語名**: ステータスバー表示
- **説明**: ステータスバーにTODO項目の数を表示します
- **デフォルト値**: `"none"`
- **型**: `string`
- **選択肢**: `"none"`, `"total"`, `"top three"`, `"tags"`, `"current file"`

### `todo-tree.general.tagGroups`
- **日本語名**: タググループ
- **説明**: 関連するタグをグループ化します
- **デフォルト値**: `{}`
- **型**: `object`

### `todo-tree.general.tags`
- **日本語名**: タグ一覧
- **説明**: 検索対象のTODOタグを定義します
- **デフォルト値**: `["TODO", "FIXME", "BUG", "HACK", "NOTE", "XXX"]`
- **型**: `array`

## ツリー表示設定 (Tree Settings)

### `todo-tree.tree.autoRefresh`
- **日本語名**: 自動更新
- **説明**: ファイル保存時にツリーを自動更新します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.buttons.export`
- **日本語名**: エクスポートボタン表示
- **説明**: ツリービューにエクスポートボタンを表示します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.buttons.groupByTag`
- **日本語名**: タグ別グループ化ボタン表示
- **説明**: タグ別グループ化ボタンを表示します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.buttons.refresh`
- **日本語名**: 更新ボタン表示
- **説明**: ツリー更新ボタンを表示します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.buttons.reveal`
- **日本語名**: 表示ボタン
- **説明**: ファイルエクスプローラーで表示するボタンを表示します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.disableCompactFolders`
- **日本語名**: コンパクトフォルダを無効化
- **説明**: VS Codeのコンパクトフォルダ機能を無効化します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.expanded`
- **日本語名**: ツリー展開状態
- **説明**: 起動時のツリー展開状態を設定します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.filterCaseSensitive`
- **日本語名**: フィルタの大文字小文字区別
- **説明**: フィルタ検索で大文字小文字を区別します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.flat`
- **日本語名**: フラット表示
- **説明**: ツリーをフラット（階層なし）で表示します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.groupedByTag`
- **日本語名**: タグ別グループ化
- **説明**: TODO項目をタグ別にグループ化して表示します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.hideIconsWhenGroupedByTag`
- **日本語名**: タググループ時アイコン非表示
- **説明**: タグ別グループ表示時にアイコンを非表示にします
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.hideTreeWhenEmpty`
- **日本語名**: 空の時ツリー非表示
- **説明**: TODO項目がない時にツリー全体を非表示にします
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.labelFormat`
- **日本語名**: ラベル形式
- **説明**: ツリー項目のラベル表示形式を設定します
- **デフォルト値**: `"${tag} ${after}"`
- **型**: `string`

### `todo-tree.tree.scanMode`
- **日本語名**: スキャンモード
- **説明**: ファイルスキャン方式を設定します
- **デフォルト値**: `"workspace"`
- **型**: `string`
- **選択肢**: `"workspace"`, `"open files"`, `"current file"`

### `todo-tree.tree.showBadges`
- **日本語名**: バッジ表示
- **説明**: ツリー項目にバッジ（カウント）を表示します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.showCountsInTree`
- **日本語名**: ツリー内カウント表示
- **説明**: フォルダ内のTODO数をツリーに表示します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.showScanModeButton`
- **日本語名**: スキャンモードボタン表示
- **説明**: スキャンモード切り替えボタンを表示します
- **デフォルト値**: `false`
- **型**: `boolean`

### `todo-tree.tree.sort`
- **日本語名**: ソート順
- **説明**: ツリー項目のソート方法を設定します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.tree.tooltipFormat`
- **日本語名**: ツールチップ形式
- **説明**: ホバー時のツールチップ表示形式を設定します
- **デフォルト値**: `"${filepath}, line ${line}"`
- **型**: `string`

### `todo-tree.tree.trackFile`
- **日本語名**: ファイル追跡
- **説明**: 現在開いているファイルのTODOをハイライトします
- **デフォルト値**: `true`
- **型**: `boolean`

## ハイライト設定 (Highlight Settings)

### `todo-tree.highlights.customHighlight`
- **日本語名**: カスタムハイライト
- **説明**: 各タグのカスタムハイライト設定を定義します
- **デフォルト値**: `{}`
- **型**: `object`

#### ハイライトオブジェクトのプロパティ

各タグのハイライト設定オブジェクトで使用できるプロパティ：

##### `type`
- **説明**: ハイライトの種類を指定
- **型**: `string`
- **選択肢**:
  - `"text"`: テキストのみハイライト
  - `"text-and-comment"`: テキストとコメント全体をハイライト
  - `"tag"`: タグ部分のみハイライト
  - `"tag-and-comment"`: タグとコメント全体をハイライト
  - `"line"`: 行全体をハイライト
  - `"whole-line"`: 行全体（左端から右端まで）をハイライト

##### `foreground`
- **説明**: 前景色（文字色）を指定
- **型**: `string`
- **例**: `"#FF0000"`, `"red"`, `"rgb(255, 0, 0)"`

##### `background`
- **説明**: 背景色を指定
- **型**: `string`
- **例**: `"#FF000020"`, `"rgba(255, 0, 0, 0.2)"`

##### `opacity`
- **説明**: 透明度を指定（0.0 - 1.0）
- **型**: `number`
- **例**: `0.5`

##### `fontWeight`
- **説明**: フォントの太さを指定
- **型**: `string`
- **選択肢**: `"normal"`, `"bold"`, `"100"` - `"900"`

##### `fontStyle`
- **説明**: フォントスタイルを指定
- **型**: `string`
- **選択肢**: `"normal"`, `"italic"`, `"oblique"`

##### `textDecoration`
- **説明**: テキスト装飾を指定
- **型**: `string`
- **選択肢**: `"none"`, `"underline"`, `"overline"`, `"line-through"`

##### `borderRadius`
- **説明**: 角の丸みを指定（px単位）
- **型**: `string`
- **例**: `"3px"`, `"50%"`

##### `icon`
- **説明**: ツリービューで表示するアイコンを指定
- **型**: `string`
- **例**: `"check"`, `"bug"`, `"alert"`, `"info"`, `"question"`
- **利用可能なアイコン**: [Codicons](https://microsoft.github.io/vscode-codicons/dist/codicon.html)を参照

##### `iconColour`
- **説明**: アイコンの色を指定
- **型**: `string`
- **例**: `"#FF0000"`, `"blue"`

##### `gutterIcon`
- **説明**: エディタのガター（行番号の横）に表示するアイコン
- **型**: `boolean` または `string`
- **例**: `true`, `"check"`, `"bug"`

##### `rulerColour`
- **説明**: エディタの右端ルーラーに表示する色
- **型**: `string`
- **例**: `"#FF0000"`

##### `rulerLane`
- **説明**: ルーラーの表示位置
- **型**: `string`
- **選択肢**: `"left"`, `"center"`, `"right"`, `"full"`

### `todo-tree.highlights.defaultHighlight`
- **日本語名**: デフォルトハイライト
- **説明**: デフォルトのハイライト設定を定義します（上記のプロパティが使用可能）
- **デフォルト値**: 
```json
{
  "type": "text",
  "foreground": "#FFCC00",
  "background": "#FFCC0020",
  "opacity": 1.0,
  "fontWeight": "normal",
  "fontStyle": "normal",
  "textDecoration": "none",
  "borderRadius": "3px"
}
```
- **型**: `object`

### `todo-tree.highlights.enabled`
- **日本語名**: ハイライト有効化
- **説明**: エディタ内でのTODOハイライトを有効化します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.highlights.highlightDelay`
- **日本語名**: ハイライト遅延
- **説明**: ハイライト更新の遅延時間（ミリ秒）を設定します
- **デフォルト値**: `500`
- **型**: `number`

### `todo-tree.highlights.useColourScheme`
- **日本語名**: カラースキーム使用
- **説明**: VS Codeのカラースキームを使用してハイライトします
- **デフォルト値**: `false`
- **型**: `boolean`

## 正規表現設定 (Regex Settings)

### `todo-tree.regex.regex`
- **日本語名**: 正規表現パターン
- **説明**: TODOを検索する正規表現パターンを設定します
- **デフォルト値**: `"((//|#|<!--|;|/\\*|^|^[ \\t]*(-|\\d+.))\\s*($TAGS)|^\\s*- \\[ \\])"`
- **型**: `string`

### `todo-tree.regex.regexCaseSensitive`
- **日本語名**: 正規表現大文字小文字区別
- **説明**: 正規表現検索で大文字小文字を区別します
- **デフォルト値**: `true`
- **型**: `boolean`

## フィルタリング設定 (Filtering Settings)

### `todo-tree.filtering.excludeGlobs`
- **日本語名**: 除外パターン
- **説明**: 検索から除外するファイル/フォルダのglobパターンを設定します
- **デフォルト値**: `[]`
- **型**: `array`

### `todo-tree.filtering.includeGlobs`
- **日本語名**: 包含パターン
- **説明**: 検索に含めるファイル/フォルダのglobパターンを設定します
- **デフォルト値**: `[]`
- **型**: `array`

### `todo-tree.filtering.includedWorkspaces`
- **日本語名**: 含まれるワークスペース
- **説明**: マルチルートワークスペースで検索対象とするワークスペースを指定します
- **デフォルト値**: `[]`
- **型**: `array`

### `todo-tree.filtering.passGlobsToRipgrep`
- **日本語名**: ripgrepにglobを渡す
- **説明**: ripgrepコマンドにglobパターンを直接渡します
- **デフォルト値**: `true`
- **型**: `boolean`

### `todo-tree.filtering.useBuiltInExcludes`
- **日本語名**: 組み込み除外パターン使用
- **説明**: VS Codeの組み込み除外パターンを使用します
- **デフォルト値**: `"file explorer"`
- **型**: `string`
- **選択肢**: `"none"`, `"file explorer"`, `"search"`

## リップグレップ設定 (Ripgrep Settings)

### `todo-tree.ripgrep.ripgrep`
- **日本語名**: ripgrepパス
- **説明**: ripgrep実行ファイルのパスを指定します
- **デフォルト値**: `""`
- **型**: `string`

### `todo-tree.ripgrep.ripgrepArgs`
- **日本語名**: ripgrep引数
- **説明**: ripgrepコマンドの追加引数を設定します
- **デフォルト値**: `"--max-columns=1000"`
- **型**: `string`

### `todo-tree.ripgrep.ripgrepMaxBuffer`
- **日本語名**: ripgrep最大バッファ
- **説明**: ripgrepの最大バッファサイズを設定します
- **デフォルト値**: `200`
- **型**: `number`

## 使用例

### 基本設定例

```json
{
  "todo-tree.general.tags": [
    "TODO",
    "FIXME", 
    "BUG",
    "HACK",
    "NOTE",
    "XXX",
    "OPTIMIZE",
    "REVIEW"
  ],
  "todo-tree.tree.showScanModeButton": true,
  "todo-tree.tree.flat": false,
  "todo-tree.tree.groupedByTag": true,
  "todo-tree.highlights.enabled": true
}
```

### 詳細なカスタムハイライト設定例

```json
{
  "todo-tree.highlights.customHighlight": {
    "TODO": {
      "icon": "check",
      "type": "line",
      "foreground": "#FFCC00",
      "background": "#FFCC0015",
      "fontWeight": "bold",
      "iconColour": "#FFCC00",
      "gutterIcon": true,
      "rulerColour": "#FFCC00",
      "rulerLane": "right"
    },
    "FIXME": {
      "icon": "bug",
      "type": "whole-line", 
      "foreground": "#FF6B6B",
      "background": "#FF6B6B20",
      "fontWeight": "bold",
      "textDecoration": "underline",
      "iconColour": "#FF6B6B",
      "gutterIcon": "bug",
      "rulerColour": "#FF6B6B"
    },
    "BUG": {
      "icon": "alert",
      "type": "text-and-comment",
      "foreground": "#FF0000",
      "background": "#FF000025",
      "fontWeight": "bold",
      "fontStyle": "italic",
      "borderRadius": "5px",
      "iconColour": "#FF0000"
    },
    "NOTE": {
      "icon": "info",
      "type": "tag-and-comment",
      "foreground": "#4A90E2",
      "background": "#4A90E220",
      "fontStyle": "italic",
      "iconColour": "#4A90E2",
      "opacity": 0.8
    },
    "HACK": {
      "icon": "flame",
      "type": "text",
      "foreground": "#FFA500",
      "background": "#FFA50030",
      "fontWeight": "bold",
      "textDecoration": "line-through",
      "iconColour": "#FFA500"
    },
    "OPTIMIZE": {
      "icon": "zap",
      "type": "tag",
      "foreground": "#9B59B6",
      "background": "#9B59B625",
      "fontWeight": "normal",
      "iconColour": "#9B59B6"
    },
    "REVIEW": {
      "icon": "eye",
      "type": "line",
      "foreground": "#2ECC71",
      "background": "#2ECC7115",
      "fontStyle": "oblique",
      "iconColour": "#2ECC71",
      "borderRadius": "3px"
    }
  }
}
```

### ハイライトタイプの違い

```javascript
// type: "text" - テキスト部分のみ
// TODO: この部分だけハイライト

// type: "text-and-comment" - テキストとコメント全体
// TODO: この行全体がハイライト

// type: "tag" - タグ部分のみ
// TODO: この部分だけハイライト

// type: "tag-and-comment" - タグとコメント全体  
// TODO: この行全体がハイライト

// type: "line" - 行全体（コード部分まで）
function example() { // TODO: 行全体がハイライト

// type: "whole-line" - 行全体（左端から右端まで）
function example() { // TODO: 画面幅全体がハイライト
```

## 注意事項

- 設定変更後は、Todo Treeパネルの更新ボタンをクリックするか、VS Codeを再起動して変更を反映してください
- 大きなプロジェクトでは、パフォーマンスを考慮して適切な除外パターンを設定することをお勧めします
- カスタムハイライトを設定する場合は、VS Codeのテーマとの互換性を確認してください