from pathlib import Path

import kagglehub

DATASET_ID = "jiashenliu/515k-hotel-reviews-data-in-europe"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_dataset(dataset_id: str = DATASET_ID, output_dir: Path = RAW_DATA_DIR) -> str:
    """Download the dataset from Kaggle into the project and return its local path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return kagglehub.dataset_download(dataset_id, output_dir=str(output_dir))
