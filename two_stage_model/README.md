# two_stage_model の説明

## 目的

このディレクトリには、`current_best.py` をもとにした 2 段階モデルの実験プログラムを置いている。

元の `current_best.py` では、`popularity` を最初から 1 つの連続値として予測していた。

一方で、`two_stage_model.py` では次のように処理を分けている。

1. まず、その曲の `popularity` が 0 か、0 より大きいかを分類する。
2. 次に、0 より大きいと考えられる曲について、具体的な `popularity` の値を回帰で予測する。

このように分けた理由は、`popularity=0` の曲が、単に人気が低い曲というだけでなく、データ上で少し特殊な曲である可能性があるためである。

## current_best.py から変えた主な点

### 1. 回帰だけでなく分類も使うようにした

`current_best.py` では、CatBoost の回帰モデルだけを使っていた。

```python
from catboost import CatBoostRegressor
```

`two_stage_model.py` では、0 か非0かを判定するために、分類モデルも使っている。

```python
from catboost import CatBoostClassifier, CatBoostRegressor
```

`CatBoostClassifier` は分類用、`CatBoostRegressor` は回帰用である。

### 2. 処理を関数に分けた

`current_best.py` は、上から順番に処理が流れる形で書かれていた。

`two_stage_model.py` では、何をしている部分か分かりやすくするために、処理を関数に分けている。

主な関数は次の通りである。

```python
def load_data(max_rows=None):
    ...
```

データを読み込み、train と test の重複を確認する。

```python
def make_features(d_train, d_test):
    ...
```

曲名、アーティスト名、ジャンル、音響特徴量などから特徴量を作る。

```python
def add_target_encoding(X_train_cv, X_valid_cv, X_test_cv, y_train_cv):
    ...
```

Fold の中で、ジャンルごとの人気度平均を特徴量として追加する。

```python
def train_two_stage_cv(...):
    ...
```

2 段階モデルの学習と検証を行う。

## 特徴量作成は基本的に current_best.py と同じ

2 段階モデルでは、モデルの構造を比較しやすくするため、特徴量作成の方針は `current_best.py` から大きく変えていない。

例えば、次のような処理は引き続き使っている。

- 曲名を TF-IDF と SVD で数値化する。
- アーティスト名を TF-IDF と SVD で数値化する。
- アルバムやアーティストの出現回数を特徴量にする。
- 音響特徴量を組み合わせる。
- ジャンルやアーティストごとの平均・標準偏差を特徴量にする。
- K-means で音響的に似た曲のクラスタを作る。

そのため、この実験では「特徴量を変えた効果」ではなく、「モデルを 2 段階に分けた効果」を比較しやすい。

## よく変更する場所

他の人が実験しやすいように、変更されやすい値は `two_stage_model.py` の上部にまとめている。

### SVD の次元数

曲名とアーティスト名を TF-IDF で数値化したあと、SVD で次元圧縮している。

```python
TRACK_NAME_SVD_COMPONENTS = 5
ARTISTS_SVD_COMPONENTS = 10
```

曲名やアーティスト名から作る特徴量を増やしたい場合は、この値を大きくする。

ただし、大きくしすぎると特徴量が増え、学習時間が長くなったり過学習しやすくなったりする可能性がある。

### レアカテゴリの閾値

出現回数が少ないアルバム名やアーティスト名は、`Rare_Album` や `Rare_Artist` にまとめている。

```python
RARE_CATEGORY_THRESHOLD = 3
```

この値を大きくすると、より多くのカテゴリが `Rare` にまとめられる。

この値を小さくすると、細かいアルバム名やアーティスト名を残しやすくなる。

### K-means のクラスタ数

音響特徴量をもとに、曲をクラスタに分けている。

```python
N_AUDIO_CLUSTERS = 10
```

音響的な分類を細かくしたい場合は、この値を大きくする。

### 集約特徴量に使う列

ジャンルやメインアーティストごとの平均・標準偏差を作る対象は、次の定数で指定している。

```python
GROUP_KEYS = ["track_genre", "main_artist"]
AGG_TARGET_VALUES = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "acousticness",
    "duration_ms",
]
```

例えば、`explicit` ごとの統計量を作りたい場合は、`GROUP_KEYS` に追加することを検討できる。

ただし、グループを増やしすぎると特徴量が増え、処理時間も長くなる。

### カテゴリ特徴量

CatBoost にカテゴリとして扱わせる列は、次の定数で指定している。

```python
BASE_CATEGORICAL_FEATURES = [
    "artists",
    "main_artist",
    "album_name",
    "explicit",
    "track_genre",
    "audio_cluster",
    "mode",
    "key",
    "time_signature",
]
```

新しくカテゴリ特徴量を追加した場合は、このリストにも追加する必要がある。

### CatBoost の共通パラメータ

分類モデルと回帰モデルで共通して使う CatBoost のパラメータは、次の辞書にまとめている。

```python
CATBOOST_BASE_PARAMS = {
    "learning_rate": 0.06530709863955916,
    "depth": 9,
    "l2_leaf_reg": 1.6325232667516456,
    "random_strength": 6.95927721754944,
    "bagging_temperature": 0.3565475988102846,
    "verbose": 300,
    "thread_count": -1,
}
```

モデルの強さや学習時間を調整したい場合は、まずこの部分を見るとよい。

### 2 段階モデルの閾値候補

学習後に比較する閾値候補は、次の定数で指定している。

```python
THRESHOLD_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6]
```

この値を変えると、表示される閾値別 CV MSE の候補を変更できる。

## 0/非0の分類をしている部分

分類用の正解ラベルは、次の行で作っている。

```python
y_is_positive = (y_train > 0).astype(int)
```

これは、`popularity > 0` なら `1`、`popularity == 0` なら `0` という意味である。

各 Fold の中では、訓練用と検証用に分けて使う。

```python
y_train_cls = y_is_positive[train_idx]
y_valid_cls = y_is_positive[valid_idx]
```

その後、分類モデルを作って学習する。

```python
clf = make_classifier(iterations, random_seed=fold)
clf.fit(
    X_train_cv,
    y_train_cls,
    cat_features=categorical_features,
    eval_set=(X_valid_cv, y_valid_cls),
    early_stopping_rounds=early_stopping_rounds,
)
```

ここで学習しているのは、具体的な `popularity` の値ではなく、「0 か非0か」である。

学習後、検証データとテストデータに対して、非0である確率を予測する。

```python
valid_positive_prob = clf.predict_proba(X_valid_cv)[:, 1]
test_positive_prob = clf.predict_proba(X_test_cv)[:, 1]
```

`valid_positive_prob` は、「この曲は `popularity > 0` である可能性」を表す値である。

例えば、この値が `0.95` なら、モデルはその曲をかなり非0らしいと判断している。

## 非0データだけで回帰している部分

回帰モデルでは、`popularity > 0` の訓練データだけを使う。

そのため、まず次のように非0の行だけを選んでいる。

```python
positive_mask = y_train_cv > 0
X_train_reg = X_train_cv.loc[positive_mask].copy()
y_train_reg = y_train_cv[positive_mask]
```

`positive_mask` は、訓練 Fold の中で `popularity > 0` の行を表す。

この `X_train_reg` と `y_train_reg` を使って、回帰モデルを学習する。

```python
reg = make_regressor(iterations, random_seed=fold)
reg.fit(
    X_train_reg,
    y_train_reg,
    cat_features=categorical_features,
    eval_set=(X_valid_cv, y_valid_cv),
    early_stopping_rounds=early_stopping_rounds,
)
```

ここで学習しているのは、非0の曲に対する具体的な `popularity` の値である。

検証データとテストデータに対しては、次のように回帰予測を行う。

```python
valid_reg_pred = np.clip(reg.predict(X_valid_cv), 0, 100)
test_reg_pred = np.clip(reg.predict(X_test_cv), 0, 100)
```

`np.clip(..., 0, 100)` は、予測値を 0 から 100 の範囲に収めるための処理である。

## 分類と回帰の結果を組み合わせる部分

分類モデルは「非0である確率」を出し、回帰モデルは「具体的な人気度」を出す。

この 2 つを次のように組み合わせて、最終予測を作っている。

```python
fold_preds = np.where(valid_positive_prob < threshold, 0.0, valid_reg_pred)
fold_preds = np.clip(fold_preds, 0, 100)
```

意味は次の通りである。

- `valid_positive_prob < threshold` の曲は 0 と予測する。
- それ以外の曲は、回帰モデルの予測値を使う。

例えば `threshold=0.5` の場合、非0である確率が 0.5 未満なら 0 とする。

テストデータの予測でも同じ考え方を使っている。

```python
test_preds = np.where(test_positive_probs < threshold, 0.0, test_reg_preds)
test_preds = np.clip(test_preds, 0, 100)
```

## 評価で出している値

各 Fold では、次のような値を表示する。

```python
print(f"Fold MSE: {fold_mse:.4f}")
print(
    "Fold classifier "
    f"accuracy={fold_acc:.4f}, precision={fold_precision:.4f}, "
    f"recall={fold_recall:.4f}"
)
```

それぞれの意味は次の通りである。

- `Fold MSE`
  - 分類と回帰を組み合わせた最終予測の誤差である。
  - 小さいほど良い。
- `accuracy`
  - 0 か非0かの分類が、全体でどれくらい当たったかを表す。
- `precision`
  - 非0だと予測した曲のうち、本当に非0だった割合を表す。
- `recall`
  - 本当に非0だった曲のうち、非0だと正しく拾えた割合を表す。

このモデルでは、特に `recall` が重要である。

本当は `popularity=40` の曲を 0 と判断してしまうと、予測が大きく外れるためである。

## 閾値別の比較

`threshold=0.5` だけが正しいとは限らない。

そこで、学習後に次のような閾値ごとの MSE も表示している。

```python
for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
    preds = np.where(oof_positive_probs < th, 0.0, oof_reg_preds)
    preds = np.clip(preds, 0, 100)
    mse = mean_squared_error(y_train, preds)
    recall = recall_score(y_is_positive, oof_positive_probs >= th, zero_division=0)
```

閾値を高くすると、0 と判定される曲が増える。

その結果、0 の曲は当てやすくなる可能性があるが、本当は非0の曲まで 0 と判定してしまう危険も増える。

そのため、MSE と recall の両方を見る必要がある。

## 確率掛け合わせ方式

閾値で 0 にする方法とは別に、確率を回帰予測に掛ける方法も試している。

```python
prob_mul_preds = np.clip(oof_positive_probs * oof_reg_preds, 0, 100)
prob_mul_mse = mean_squared_error(y_train, prob_mul_preds)
```

これは、非0らしさが低い曲ほど予測値を小さくする方法である。

ただし、非0の曲に対しても予測値が小さくなりすぎる場合があるため、必ず良くなるとは限らない。

## current_best.py にあった上書き処理について

`current_best.py` には、test の `track_id` が train に存在する場合、train 側の `popularity` で予測値を上書きする処理があった。

考え方としては次のような処理である。

```python
if t_id in leak_dict:
    final_preds[i] = leak_dict[t_id]
```

しかし、`two_stage_model.py` ではこの上書き処理は使っていない。

train と test の重複数は表示しているが、予測値を正解値で上書きすることはしていない。

そのため、`two_stage_model/y_pred_two_stage.csv` は、2 段階モデルによる予測結果である。

## 実行方法
以下のディレクトリ構造を用意。
project/                
├── data/
│   ├── train.csv          
│   └── test.csv           
└── two_stage_model/
    └── two_stage_model.py

フル実行は次の通りである。

```bash
python two_stage_model/two_stage_model.py
```

条件を付ける、動作確認の場合には以下のように実行できる。
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python two_stage_model/two_stage_model.py --max-rows 1000 --folds 2 --iterations 5
```


## Google Colab で実行する方法

まず、必要なライブラリをインストールする。

```python
!pip install catboost pandas numpy scikit-learn
```

次に、Colab 上で次のようなファイル構成になるようにする。

```text
project/
├── data/
│   ├── train.csv
│   └── test.csv
└── two_stage_model/
    ├── two_stage_model.py
    └── (README.md)
```

Google Drive にファイルを置く場合は、最初に Drive をマウントする。

```python
from google.colab import drive
drive.mount('/content/drive')
```

その後、`project` ディレクトリに移動する。

例えば、Google Drive の `MyDrive` 直下に `project` ディレクトリを置いた場合は、次のように移動する。

```python
%cd /content/drive/MyDrive/prject
```

動作確認だけを軽く行う場合は、行数や学習回数を減らして実行する。

```python
!python two_stage_model/two_stage_model.py --max-rows 1000 --folds 2 --iterations 5
```

最後まで動くことを確認できたら、通常実行を行う。

```python
!python two_stage_model/two_stage_model.py
```

閾値を変えて実行したい場合は、`--threshold` を指定する。

```python
!python two_stage_model/two_stage_model.py --threshold 0.3
```

Colab で実行した場合も、予測結果は次のファイルに保存される。

```text
two_stage_model/y_pred_two_stage.csv
```

## 出力ファイル

通常実行すると、次のファイルに予測結果が保存される。

```text
two_stage_model/y_pred_two_stage.csv
```
