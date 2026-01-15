from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from src.utils.config import load_config


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load raw churn dataset."""
    raw_path = Path(config["data"]["raw_path"])
    churn_file = config["data"]["churn_file"]
    file_path = raw_path / churn_file
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data not found: {file_path}")
    return pd.read_csv(file_path)


def main() -> None:
    config = load_config()
    df = load_raw_data(config)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    main()
