# Character Template Cropper

鉄拳8の対戦画面上部にあるキャラクターHUD（顔アイコンとローマ字名）を、左右別の認識用PNGとして安全に切り出す補助ツールです。既存の自動判定処理、CSV、学習モデルには接続していません。

## 必要環境と起動

Python、OpenCV、NumPy、標準ライブラリの tkinter が必要です。リポジトリ直下で起動します。

```powershell
python tools/character_template_cropper.py
```

別の設定を試す場合は `--config path/to/config.json` を指定できます。

## 基本操作

`Open image` は1枚、`Open folder` は PNG/JPG/JPEG をファイル名順に読み込みます。`|<`、`<`、`>`、`>|` で先頭・前・次・末尾へ移動します。未保存のROIまたは名前がある場合は破棄確認が表示されます。

画像上を左ドラッグすると新しいROIを選択できます。枠は黒い縁取りと、左が水色・右がオレンジの線で表示されます。右パネルのプレビューは保存される元画像ピクセルを表示します。

`Left` / `Right` で保存する側を選びます。左右はそれぞれ独立したROIを保持します。`Left → Right` と `Right → Left` は座標をそのままコピーし、`Mirror to other side` は画像中央を基準に水平反転した位置を反対側へ設定します。コピー後も自由に微調整できます。

## キーボード微調整

キャンバス上で次を使用します。

- 矢印: ROIを1px移動
- Shift + 矢印: ROIを10px移動
- Ctrl + 左右矢印: 幅を1px変更
- Ctrl + 上下矢印: 高さを1px変更
- Ctrl + Shift + 矢印: 幅または高さを10px変更

ROIは常に画像内へ補正され、幅・高さは最低1pxです。

## 表示と名前

`-`、`+`、`100%`、`Fit` で表示倍率を変更できます。倍率は表示だけに使われ、ROI座標と保存PNGは常に元画像基準です。`color` / `grayscale` は確認表示だけを切り替え、保存は常にカラーPNGです。

ファイルを開くと、たとえば `reina_right_001.png` から `reina` を仮入力します。保存前に修正可能です。名前は前後空白を除去し、小文字化し、空白や使用できない文字を `_` に正規化します。空の名前は保存できません。

## 保存先と上書き

通常は次へ保存します。左右はフォルダで分けるため、ファイル名に `_left` / `_right` は付けません。

```text
templates/characters/left/<character_name>.png
templates/characters/right/<character_name>.png
```

`Choose output folder` で一時的にルートを変更できます。リポジトリ内なら相対パスとして扱います。この変更は `Save ROI settings` を押したときだけ設定へ保存されます。

同名PNGがある場合、`Save template` は上書き確認を表示します。キャンセル時は既存ファイルを変更しません。画面には `unsaved` または `same-name file exists` が表示され、保存後に更新されます。

## 一括切り出し

フォルダを開き、左右とROIを確認して `Batch crop current folder` を押します。実行前に入力名と出力名を一覧表示します。名前を推定できない画像や、同じ出力名になる画像は拒否されます。同名ファイルがある場合は、全体を「上書き」「スキップ」「キャンセル」から選択します。意図しないファイルを削除する処理はありません。

## ROI設定

`config/character_roi.json` に次の形式で保存します。ファイルがない、壊れている、または値が不正な場合はクラッシュせず既定値で起動し、ステータス欄に結果を表示します。保存は一時ファイルからの置換で行います。パスは可能な限りリポジトリ基準の相対パスを推奨します。

```json
{
  "version": 1,
  "source_resolution": {"width": 1920, "height": 1080},
  "roi": {
    "left": {"x": 40, "y": 30, "width": 260, "height": 72},
    "right": {"x": 1620, "y": 30, "width": 260, "height": 72}
  },
  "display": {"zoom": 1.0, "mode": "color"},
  "paths": {
    "source_root": "template_sources",
    "output_root": "templates/characters"
  }
}
```

## 推奨する入力画像

- 解像度とHUD位置が固定されている
- HUDが隠れておらず、演出・モーションブラー・圧縮ノイズが少ない
- 顔アイコンとローマ字名が完全に含まれる
- 左右それぞれ、実際の照明や背景差を確認できる

トラブル時はまずステータス欄、画像解像度、選択中の side、ROI値、出力ルートを確認してください。設定がおかしい場合は設定ファイルを退避すると既定値で起動します。
