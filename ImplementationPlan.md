# 開発計画

##　現状把握
`current_best.py` は、楽曲の情報から `popularity` を予測するための機械学習プログラムである。中心となる考え方は、単純な音響特徴量だけで予測するのではなく、曲名、アーティスト名、アルバム名、ジャンル、音響的な傾向などをできるだけ特徴量として取り出し、それを CatBoost という勾配ブースティング系の回帰モデルに学習させる、というものである。

大まかな流れは次の通りである。

1. `train.csv` と `test.csv` を読み込む。
2. 訓練データとテストデータに同じ曲が含まれていないか確認する。
3. 曲名やアーティスト名などの文字列情報から特徴量を作る。
4. 音響特徴量を組み合わせたり、ジャンルごとの平均値などを作ったりする。
5. CatBoostRegressor を 5-Fold 交差検証で学習する。
6. テストデータを予測し、予測値を 0 から 100 の範囲に収める。
7. 最後に、訓練データとテストデータで `track_id` が一致する曲については、訓練データの正解値で予測値を上書きする。

全体としては、「モデル自体を複雑に作り込む」というよりも、「予測に役立ちそうな情報を多方面から特徴量として追加し、CatBoost に任せる」という方針で書かれている。


## 参考　特徴量重要度
```bash
--- 特徴量重要度 (上位25個) ---
                         Feature  Importance
                      album_name   30.678741
                     track_genre    9.522271
                     album_count    6.305771
                     main_artist    4.999196
                  te_track_genre    3.641664
                         artists    1.428160
                track_name_svd_1    1.160598
                track_name_svd_0    1.030827
                   audio_cluster    1.016763
                instrumentalness    0.939483
                     speechiness    0.925667
agg_track_genre_duration_ms_mean    0.921418
    agg_track_genre_loudness_std    0.911615
               main_artist_count    0.899543
                   artists_svd_4    0.860590
agg_track_genre_danceability_std    0.856410
 agg_main_artist_duration_ms_std    0.849509
    agg_main_artist_loudness_std    0.848685
                    danceability    0.841542
 agg_track_genre_duration_ms_std    0.841539
                   artists_svd_1    0.839408
                   artists_svd_9    0.839061
      agg_track_genre_tempo_mean    0.837939
                    duration_sec    0.836789
       agg_track_genre_tempo_std    0.819072
```
## 課題点
- [x] 訓練データとテストデータで `track-id` が一致する曲の予測値を上書きしている。
-> 純粋な予測というよりもデータ分割の性質を利用した`裏ワザ`に近い。自分の環境だと9889件のデータが上書きされており、全体の約1/4。予測力を測るという観点では適切でない可能性がある。

- [ ] 特徴量重要度を見ると、album_nameが突出して高いことが分かる。これは、モデルが「特定のアルバム・アーティストに対応する popularity の傾向」を重視していることを意味し、交差検証などで同じ楽曲・アーティストが跨っている場合は高い精度を誇るが、未知アーティストや楽曲に対しての一般性を持つかは分からない。
-> 今回のコンペの内容次第。テストデータが訓練データと跨っているならむしろ有利か。アーティストや曲の既知性に関連する特徴量を強化すれば、よりよいデータが得られるのか検証したい


## 開発方針
- [x] `popularity` が0のデータも多くあるので、まずは「`popularity` が0か、それ以外か」という分類タスクにかける。その後0以外と分類された楽曲に対して回帰タスクを回す、というモデルを作成し、性能がどう変わるか検証。　現状では、「1未満を0、それ以外は1~100」とデータをクリップしている。
->済。`two-stage-model` 参照。
- [ ] 有用な特徴量を捜索する。実験では `Album_name` が非常に重要そうであり、アルバムやアーティストデータは特徴量探索に重要なのではないか。また、`artist` よりも `main_artist` の方が高い重要性を示していた。特徴量から何か新たに作れるか。ターゲットエンコーディングの対象も変更、追加などしてみる。
```bash
     - main_artist
      - album_name
      - key
      - time_signature
      - explicit
      - track_genre × explicit
```
などにできるかも。

- [x] `album_name`や`main_artist`を特徴量から除いてみる。<- 実験したもののたいして意味はなし。 `album_name` や `track_genre` が重要であることは再確認された。

and more...
