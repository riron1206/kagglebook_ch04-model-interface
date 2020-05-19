### 「Kaggleで勝つデータ分析の技術」の第4章「分析コンペ用のクラスやフォルダの構成」サンプルコード(https://github.com/ghmagazine/kagglebook/tree/master/ch04-model-interface)

Kaggleの[Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge/)
のデータを入力として、xgboostおよびtensorflow.kerasによる学習・予測の一連の流れが行えるようにしています。  
なお、パラメータや手法はhttps://github.com/puyokw/kaggle_Otto/ を参考にしました。

以下の手順で実行することができます。

1. [データ](https://www.kaggle.com/c/otto-group-product-classification-challenge/data)をダウンロードし、`input`フォルダに保存して下さい（ダウンロード済み）。
2. `code`フォルダを作業フォルダとし、```python run.py```を実行して下さい。 

#### ディレクトリ構成
```bash
input       # train.csvなどの入力ファイル入れるディレクトリ
code        # 計算用コード入れるディレクトリ
notebook    # Jupyter Notebook用ディレクトリ
model       # モデルや特徴量を保存するディレクトリ
submission  # 提出用ファイルを保存するディレクトリ
```