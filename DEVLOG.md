# tekken8_auto_tracker

## 現在の進捗

### 完了

* Python環境構築
* venv構築
* requirements.txt導入
* OpenCV導入
* MSS導入
* メインモニターのスクリーンショット取得
* WINテンプレート作成
* LOSEテンプレート作成
* テンプレートマッチング実装
* WIN判定成功
* LOSE判定成功
* リアルタイム監視ループ実装

### 現在の動作

* 1秒ごとに画面を取得
* detector.pyでWIN/LOSE判定
* WIN/LOSEが存在しない場合はNONE
* Ctrl+Cで停止可能

## 次回作業

### 優先度高

* 同じWIN/LOSEを連続検出しない機能
* 検出時のみ表示する仕組み

### 優先度中

* JSON戦績保存
* 勝利数集計
* 敗北数集計

### 将来予定

* 鉄拳8ウィンドウ自動検出
* キャラクター判定
* ランク情報取得
* GUI化

## 現在の構成

tekken8_auto_tracker/

* capture.py
* detector.py
* requirements.txt
* templates/

  * win.png
  * lose.png
* screenshots/
* .venv/

## 備考

現在は静止画判定を卒業し、リアルタイム監視段階へ移行済み。
次のマイルストーンは「勝敗イベントを1回だけ検出すること」。
