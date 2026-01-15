from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.make_dataset import load_raw_data
from src.features.feature_engineering import TARGET, preprocess_data
from src.utils.config import load_config


def _align_feature_columns(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    extra_cols = [c for c in df.columns if c not in feature_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    return df[feature_columns]


def _evaluate_k_range(X: pd.DataFrame, k_values: range) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": score, "inertia": model.inertia_})
    return pd.DataFrame(rows)


def run_segmentation(config: Dict[str, Any]) -> None:
    processed_path = Path(config["data"]["processed_path"])
    models_path = Path("models")
    reports_path = Path("reports/figures")
    reports_path.mkdir(parents=True, exist_ok=True)

    scaler = joblib.load(processed_path / "scaler.pkl")
    feature_config = joblib.load(processed_path / "feature_config.pkl")

    raw_df = load_raw_data(config)
    df_features, _ = preprocess_data(raw_df, drop_ids=True)

    if TARGET in df_features.columns:
        df_features = df_features.drop(columns=[TARGET])

    df_features = _align_feature_columns(df_features, feature_config["feature_columns"])
    cols_to_scale = feature_config["cols_to_scale"]
    df_features[cols_to_scale] = scaler.transform(df_features[cols_to_scale])

    k_values = range(2, 11)
    diagnostics = _evaluate_k_range(df_features, k_values)
    diagnostics.to_csv(processed_path / "segmentation_diagnostics.csv", index=False)

    configured_k = config["clustering"].get("n_clusters")
    if configured_k:
        k_final = configured_k
    else:
        k_final = diagnostics.sort_values("silhouette", ascending=False).iloc[0]["k"]

    kmeans = KMeans(
        n_clusters=int(k_final),
        max_iter=config["clustering"]["max_iter"],
        random_state=config["clustering"]["random_state"],
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(df_features)

    segmented_df = raw_df.copy()
    segmented_df["Segment"] = cluster_labels

    segment_counts = segmented_df["Segment"].value_counts().sort_index()
    print("Segment sizes:")
    print(segment_counts)

    numeric_cols = segmented_df.select_dtypes(include="number").columns.tolist()
    exclude_cols = {TARGET, "RowNumber", "CustomerId"}
    profile_cols = [c for c in numeric_cols if c not in exclude_cols]
    segment_profile = segmented_df.groupby("Segment")[profile_cols].mean().round(3)

    segmented_df.to_csv(processed_path / "segmentation_labels.csv", index=False)
    segment_profile.to_csv(processed_path / "segment_profile.csv")
    joblib.dump(kmeans, models_path / "kmeans_segmentation.pkl")

    # PCA plot for quick visualization
    pca = PCA(n_components=2, random_state=42)
    pca_df = pd.DataFrame(pca.fit_transform(df_features), columns=["PC1", "PC2"])
    pca_df["Segment"] = cluster_labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Segment", palette="tab10", s=20)
    plt.title("PCA View of Segments")
    plt.tight_layout()
    plt.savefig(reports_path / "segments_pca.png", dpi=150)
    plt.close()

    print("Saved segmentation artifacts to data/processed and models/")


def main() -> None:
    config = load_config()
    run_segmentation(config)


if __name__ == "__main__":
    main()
