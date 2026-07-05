import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EXCLUDE_FEATURES = []
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

TRACK_NAME_SVD_COMPONENTS = 5
ARTISTS_SVD_COMPONENTS = 10
RARE_CATEGORY_THRESHOLD = 3
N_AUDIO_CLUSTERS = 10
THRESHOLD_CANDIDATES = [0.2, 0.3, 0.4, 0.5, 0.6]

AUDIO_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

GROUP_KEYS = ["track_genre", "main_artist"]
AGG_TARGET_VALUES = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "acousticness",
    "duration_ms",
]

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

CATBOOST_BASE_PARAMS = {
    "learning_rate": 0.06530709863955916,
    "depth": 9,
    "l2_leaf_reg": 1.6325232667516456,
    "random_strength": 6.95927721754944,
    "bagging_temperature": 0.3565475988102846,
    "verbose": 300,
    "thread_count": -1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="popularity を 0/非0分類と非0回帰に分けて学習する実験"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=2628)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="動作確認用。指定した場合は train の先頭行だけで学習する。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "two_stage_model" / "y_pred_two_stage.csv",
    )
    return parser.parse_args()


def load_data(max_rows=None):
    print("--- データを読み込み中... ---")
    d_train = pd.read_csv(DATA_DIR / "train.csv")
    d_test = pd.read_csv(DATA_DIR / "test.csv")

    if max_rows is not None:
        d_train = d_train.head(max_rows).copy()
        print(f"--- 動作確認用に train を先頭 {len(d_train)} 行に制限しました ---")

    train_ids = set(d_train["track_id"])
    test_ids = set(d_test["track_id"])
    overlap_ids = train_ids.intersection(test_ids)

    print("--- track_id の重複チェック ---")
    print(f"Trainの固有曲数: {len(train_ids)}")
    print(f"Testの固有曲数: {len(test_ids)}")
    print(f"両方に存在する track_id の数: {len(overlap_ids)}")

    d_train_str = d_train[["track_name", "artists"]].fillna("")
    d_test_str = d_test[["track_name", "artists"]].fillna("")
    train_names = set(d_train_str["track_name"] + "_" + d_train_str["artists"])
    test_names = set(d_test_str["track_name"] + "_" + d_test_str["artists"])
    overlap_names = train_names.intersection(test_names)

    print("\n--- 曲名＆アーティスト の重複チェック ---")
    print(f"両方に存在する 曲名＆アーティスト の数: {len(overlap_names)}")
    if overlap_names:
        print("重複している曲の例:")
        print(list(overlap_names)[:5])

    return d_train, d_test


def make_features(d_train, d_test):
    for df in [d_train, d_test]:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    n_train = len(d_train)
    y_train = d_train.pop("popularity").to_numpy()
    d_all = pd.concat([d_train, d_test], axis=0).reset_index(drop=True)

    print("--- 特徴量エンジニアリングを実行中... ---")

    d_all["track_name"] = d_all["track_name"].fillna("").astype(str)
    tfidf_track = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, stop_words="english"
    )
    track_name_tfidf = tfidf_track.fit_transform(d_all["track_name"])

    svd_track = TruncatedSVD(n_components=TRACK_NAME_SVD_COMPONENTS, random_state=42)
    track_name_svd = svd_track.fit_transform(track_name_tfidf)
    for i in range(TRACK_NAME_SVD_COMPONENTS):
        d_all[f"track_name_svd_{i}"] = track_name_svd[:, i]

    d_all["is_remix"] = d_all["track_name"].str.contains(
        "Remix", case=False
    ).astype(int)
    d_all["is_live"] = d_all["track_name"].str.contains("Live", case=False).astype(
        int
    )
    d_all["is_acoustic"] = d_all["track_name"].str.contains(
        "Acoustic", case=False
    ).astype(int)
    d_all["is_feat"] = d_all["track_name"].str.contains("feat.", case=False).astype(
        int
    )

    d_all["artists"] = d_all["artists"].fillna("Unknown").astype(str)
    tfidf_art = TfidfVectorizer(analyzer="word", token_pattern=r"[^;]+", min_df=2)
    artists_tfidf = tfidf_art.fit_transform(d_all["artists"])

    svd_art = TruncatedSVD(n_components=ARTISTS_SVD_COMPONENTS, random_state=42)
    artists_svd = svd_art.fit_transform(artists_tfidf)
    for i in range(ARTISTS_SVD_COMPONENTS):
        d_all[f"artists_svd_{i}"] = artists_svd[:, i]

    d_all["num_artists"] = d_all["artists"].apply(
        lambda x: len(x.split(";")) if x != "Unknown" else 1
    )
    d_all["main_artist"] = d_all["artists"].apply(lambda x: x.split(";")[0])

    album_counts = d_all["album_name"].value_counts()
    artist_counts = d_all["main_artist"].value_counts()

    d_all["album_count"] = d_all["album_name"].map(album_counts)
    d_all["main_artist_count"] = d_all["main_artist"].map(artist_counts)
    d_all["is_single"] = (d_all["album_count"] == 1).astype(int)

    d_all["album_name"] = d_all["album_name"].apply(
        lambda x: x if album_counts.get(x, 0) > RARE_CATEGORY_THRESHOLD else "Rare_Album"
    )
    d_all["main_artist"] = d_all["main_artist"].apply(
        lambda x: x
        if artist_counts.get(x, 0) > RARE_CATEGORY_THRESHOLD
        else "Rare_Artist"
    )

    d_all["mode_valence_1"] = d_all["mode"] * d_all["valence"]
    d_all["mode_energy_1"] = d_all["mode"] * d_all["energy"]
    d_all["mode_valence_0"] = (1.0 - d_all["mode"]) * d_all["valence"]
    d_all["mode_energy_0"] = (1.0 - d_all["mode"]) * d_all["energy"]

    d_all["energy_per_loudness"] = d_all["energy"] / (
        d_all["loudness"] - d_all["loudness"].min() + 1e-6
    )
    d_all["acoutcis_loudness_gap"] = d_all["acousticness"] * d_all["loudness"]
    d_all["duration_sec"] = d_all["duration_ms"] / 1000
    d_all["dance_energy"] = d_all["danceability"] * d_all["energy"]

    for key in GROUP_KEYS:
        for val in AGG_TARGET_VALUES:
            agg = d_all.groupby(key)[val].agg(["mean", "std"]).reset_index()
            agg.columns = [key, f"agg_{key}_{val}_mean", f"agg_{key}_{val}_std"]
            d_all = pd.merge(d_all, agg, on=key, how="left")

    eps = 1e-6
    d_all["genre_loudness_hensachi"] = (
        (d_all["loudness"] - d_all["agg_track_genre_loudness_mean"])
        / (d_all["agg_track_genre_loudness_std"] + eps)
    ) * 10 + 50
    d_all["genre_energy_hensachi"] = (
        (d_all["energy"] - d_all["agg_track_genre_energy_mean"])
        / (d_all["agg_track_genre_energy_std"] + eps)
    ) * 10 + 50
    d_all["genre_duration_hensachi"] = (
        (d_all["duration_ms"] - d_all["agg_track_genre_duration_ms_mean"])
        / (d_all["agg_track_genre_duration_ms_std"] + eps)
    ) * 10 + 50

    scaler = StandardScaler()
    audio_scaled = scaler.fit_transform(d_all[AUDIO_COLS])
    kmeans = KMeans(n_clusters=N_AUDIO_CLUSTERS, random_state=0, n_init=10)
    d_all["audio_cluster"] = kmeans.fit_predict(audio_scaled)

    d_all = d_all.drop(columns=["track_id", "track_name", "duration_ms"])

    d_all["album_name"] = d_all["album_name"].fillna("Unknown")
    d_all["track_genre"] = d_all["track_genre"].fillna("Unknown")

    categorical_features = BASE_CATEGORICAL_FEATURES.copy()
    categorical_features = [col for col in categorical_features if col not in EXCLUDE_FEATURES]

    if EXCLUDE_FEATURES:
        d_all = d_all.drop(columns=EXCLUDE_FEATURES)

    for col in categorical_features:
        d_all[col] = d_all[col].astype(str)

    df_train_features = d_all.iloc[:n_train].reset_index(drop=True)
    df_test_features = d_all.iloc[n_train:].reset_index(drop=True)

    return df_train_features, df_test_features, y_train, categorical_features


def add_target_encoding(X_train_cv, X_valid_cv, X_test_cv, y_train_cv):
    X_train_cv = X_train_cv.copy()
    X_valid_cv = X_valid_cv.copy()
    X_test_cv = X_test_cv.copy()

    X_train_cv["tmp_target"] = y_train_cv
    target_mean = X_train_cv.groupby("track_genre")["tmp_target"].mean()
    global_mean = y_train_cv.mean()

    X_train_cv["te_track_genre"] = (
        X_train_cv["track_genre"].map(target_mean).fillna(global_mean)
    )
    X_valid_cv["te_track_genre"] = (
        X_valid_cv["track_genre"].map(target_mean).fillna(global_mean)
    )
    X_test_cv["te_track_genre"] = (
        X_test_cv["track_genre"].map(target_mean).fillna(global_mean)
    )

    X_train_cv.drop(columns=["tmp_target"], inplace=True)

    return X_train_cv, X_valid_cv, X_test_cv


def make_classifier(iterations, random_seed):
    return CatBoostClassifier(
        **CATBOOST_BASE_PARAMS,
        iterations=iterations,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
    )


def make_regressor(iterations, random_seed):
    return CatBoostRegressor(
        **CATBOOST_BASE_PARAMS,
        iterations=iterations,
        eval_metric="RMSE",
        random_seed=random_seed,
    )


def train_two_stage_cv(
    df_train_features,
    df_test_features,
    y_train,
    categorical_features,
    folds,
    threshold,
    iterations,
    early_stopping_rounds,
):
    print(f"--- 2段階モデルの {folds}-Fold CV を開始します ---")

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof_reg_preds = np.zeros(len(y_train))
    oof_positive_probs = np.zeros(len(y_train))
    test_reg_preds = np.zeros(len(df_test_features))
    test_positive_probs = np.zeros(len(df_test_features))
    y_is_positive = (y_train > 0).astype(int)
    reg_feature_importances = None

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df_train_features, y_train)):
        print(f"\n--- Fold {fold + 1} / {folds} ---")

        X_train_cv = df_train_features.iloc[train_idx].copy()
        X_valid_cv = df_train_features.iloc[valid_idx].copy()
        X_test_cv = df_test_features.copy()

        y_train_cv = y_train[train_idx]
        y_valid_cv = y_train[valid_idx]
        y_train_cls = y_is_positive[train_idx]
        y_valid_cls = y_is_positive[valid_idx]

        X_train_cv, X_valid_cv, X_test_cv = add_target_encoding(
            X_train_cv, X_valid_cv, X_test_cv, y_train_cv
        )

        clf = make_classifier(iterations, random_seed=fold)
        clf.fit(
            X_train_cv,
            y_train_cls,
            cat_features=categorical_features,
            eval_set=(X_valid_cv, y_valid_cls),
            early_stopping_rounds=early_stopping_rounds,
        )

        valid_positive_prob = clf.predict_proba(X_valid_cv)[:, 1]
        test_positive_prob = clf.predict_proba(X_test_cv)[:, 1]

        positive_mask = y_train_cv > 0
        X_train_reg = X_train_cv.loc[positive_mask].copy()
        y_train_reg = y_train_cv[positive_mask]

        reg = make_regressor(iterations, random_seed=fold)
        reg.fit(
            X_train_reg,
            y_train_reg,
            cat_features=categorical_features,
            eval_set=(X_valid_cv, y_valid_cv),
            early_stopping_rounds=early_stopping_rounds,
        )

        valid_reg_pred = np.clip(reg.predict(X_valid_cv), 0, 100)
        test_reg_pred = np.clip(reg.predict(X_test_cv), 0, 100)

        oof_positive_probs[valid_idx] = valid_positive_prob
        oof_reg_preds[valid_idx] = valid_reg_pred
        test_positive_probs += test_positive_prob / folds
        test_reg_preds += test_reg_pred / folds

        if reg_feature_importances is None:
            reg_feature_importances = np.zeros(len(X_train_cv.columns))
        reg_feature_importances += reg.get_feature_importance() / folds

        fold_preds = np.where(valid_positive_prob < threshold, 0.0, valid_reg_pred)
        fold_preds = np.clip(fold_preds, 0, 100)
        fold_mse = mean_squared_error(y_valid_cv, fold_preds)
        fold_acc = accuracy_score(y_valid_cls, valid_positive_prob >= threshold)
        fold_precision = precision_score(
            y_valid_cls, valid_positive_prob >= threshold, zero_division=0
        )
        fold_recall = recall_score(
            y_valid_cls, valid_positive_prob >= threshold, zero_division=0
        )
        print(f"Fold MSE: {fold_mse:.4f}")
        print(
            "Fold classifier "
            f"accuracy={fold_acc:.4f}, precision={fold_precision:.4f}, "
            f"recall={fold_recall:.4f}"
        )

    return oof_reg_preds, oof_positive_probs, test_reg_preds, test_positive_probs, reg_feature_importances


def print_cv_report(y_train, oof_reg_preds, oof_positive_probs, threshold):
    final_oof = np.where(oof_positive_probs < threshold, 0.0, oof_reg_preds)
    final_oof = np.clip(final_oof, 0, 100)

    cv_mse = mean_squared_error(y_train, final_oof)
    y_is_positive = (y_train > 0).astype(int)
    cls_pred = (oof_positive_probs >= threshold).astype(int)

    print("\n======================================")
    print(f"2段階モデル CV MSE: {cv_mse:.4f}")
    print(f"使用した閾値: {threshold:.2f}")
    print("======================================")
    print(f"popularity == 0 の件数: {(y_train == 0).sum()}")
    print(f"popularity > 0 の件数: {(y_train > 0).sum()}")
    print(f"分類 accuracy: {accuracy_score(y_is_positive, cls_pred):.4f}")
    print(
        "分類 precision: "
        f"{precision_score(y_is_positive, cls_pred, zero_division=0):.4f}"
    )
    print(
        "分類 recall: "
        f"{recall_score(y_is_positive, cls_pred, zero_division=0):.4f}"
    )
    print(f"0の曲に対する平均予測値: {final_oof[y_train == 0].mean():.4f}")
    print(
        "非0の曲に対する MSE: "
        f"{mean_squared_error(y_train[y_train > 0], final_oof[y_train > 0]):.4f}"
    )

    print("\n--- 閾値別 CV MSE ---")
    for th in THRESHOLD_CANDIDATES:
        preds = np.where(oof_positive_probs < th, 0.0, oof_reg_preds)
        preds = np.clip(preds, 0, 100)
        mse = mean_squared_error(y_train, preds)
        recall = recall_score(y_is_positive, oof_positive_probs >= th, zero_division=0)
        print(f"threshold={th:.1f}: MSE={mse:.4f}, positive_recall={recall:.4f}")

    prob_mul_preds = np.clip(oof_positive_probs * oof_reg_preds, 0, 100)
    prob_mul_mse = mean_squared_error(y_train, prob_mul_preds)
    print(f"\n確率掛け合わせ CV MSE: {prob_mul_mse:.4f}")

    return final_oof


def save_predictions(test_reg_preds, test_positive_probs, threshold, output_path):
    test_preds = np.where(test_positive_probs < threshold, 0.0, test_reg_preds)
    test_preds = np.clip(test_preds, 0, 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(X=test_preds, fname=output_path)
    print(f"\n予測結果を '{output_path}' に保存しました。")


def main():
    args = parse_args()
    d_train, d_test = load_data(max_rows=args.max_rows)
    df_train_features, df_test_features, y_train, categorical_features = make_features(
        d_train, d_test
    )

    print(f"train shape: {df_train_features.shape}")
    print(f"test shape: {df_test_features.shape}")
    print(f"popularity == 0 rate: {(y_train == 0).mean():.4f}")

    (
        oof_reg_preds,
        oof_positive_probs,
        test_reg_preds,
        test_positive_probs,
        reg_feature_importances,
    ) = train_two_stage_cv(
        df_train_features=df_train_features,
        df_test_features=df_test_features,
        y_train=y_train,
        categorical_features=categorical_features,
        folds=args.folds,
        threshold=args.threshold,
        iterations=args.iterations,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    print_cv_report(y_train, oof_reg_preds, oof_positive_probs, args.threshold)
    save_predictions(test_reg_preds, test_positive_probs, args.threshold, args.output)

    print("\n--- 回帰モデルの特徴量重要度 (上位25個) ---")
    feature_names = list(df_train_features.columns) + ["te_track_genre"]
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": reg_feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    print(importance_df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
