from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.make_dataset import load_raw_data
from src.features.feature_engineering import (
    TARGET,
    preprocess_data,
)
from src.utils.config import load_config


def _identify_binary_columns(df: pd.DataFrame) -> List[str]:
    binary_cols = []
    for col in df.columns:
        if df[col].dtype.kind in {"i", "u", "f", "b"}:
            uniques = pd.Series(df[col].dropna().unique()).tolist()
            if len(uniques) > 0 and set(uniques).issubset({0, 1}):
                binary_cols.append(col)
    return binary_cols


def build_features(config: Dict[str, Any]) -> None:
    processed_path = Path(config["data"]["processed_path"])
    processed_path.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(config)
    df_features, _ = preprocess_data(df_raw, drop_ids=True)

    if TARGET not in df_features.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    X = df_features.drop(columns=[TARGET])
    y = df_features[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)

    binary_cols = _identify_binary_columns(X_train)
    cols_to_scale = [
        c
        for c in X_train.columns
        if c not in binary_cols
        and X_train[c].dtype.kind in {"i", "u", "f"}
        and X_train[c].nunique() > 2
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    X_train_scaled.to_csv(processed_path / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(processed_path / "X_test_scaled.csv", index=False)

    feature_config = {
        "feature_columns": X_train.columns.tolist(),
        "cols_to_scale": cols_to_scale,
        "binary_columns": binary_cols,
    }
    joblib.dump(scaler, processed_path / "scaler.pkl")
    joblib.dump(feature_config, processed_path / "feature_config.pkl")

    print("Saved processed datasets to data/processed")
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Scaled columns: {len(cols_to_scale)}")


def main() -> None:
    config = load_config()
    build_features(config)


if __name__ == "__main__":
    main()
