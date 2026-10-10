"""Dataset cleaning and chronological splitting."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.preprocess import clean

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test


def split_dataset(
    df: pd.DataFrame, ratios: tuple[float, float, float] = SPLIT_RATIOS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split on `review_date`.

    Cut points are taken on date boundaries, so a day is never divided between splits
    (actual sizes are therefore close to, not exactly, the requested ratios).
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    df = df.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)
    dates = df["review_date"]
    val_start = dates.iloc[int(len(df) * ratios[0])]
    test_start = dates.iloc[int(len(df) * (ratios[0] + ratios[1]))]
    return (
        df[dates < val_start].reset_index(drop=True),
        df[(dates >= val_start) & (dates < test_start)].reset_index(drop=True),
        df[dates >= test_start].reset_index(drop=True),
    )


def prepare(csv_path: str | Path, out_dir: str | Path = PROCESSED_DIR) -> dict[str, Path]:
    """CSV -> clean/features -> split -> write parquet files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = clean(pd.read_csv(csv_path))
    paths = {}
    for name, part in zip(("train", "val", "test"), split_dataset(df)):
        paths[name] = out_dir / f"{name}.parquet"
        part.to_parquet(paths[name], index=False)
        log.info("%s: %d rows -> %s", name, len(part), paths[name])
    return paths
