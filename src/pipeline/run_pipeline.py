from __future__ import annotations

from src.features.build_features import build_features
from src.models.train_models import train_models
from src.segmentation.run_segmentation import run_segmentation
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    build_features(config)
    train_models(config)
    run_segmentation(config)


if __name__ == "__main__":
    main()
