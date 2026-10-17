"""Build everything the service needs: python main.py [--rebuild]

Steps, each skipped when its output already exists (use --rebuild to redo all):
  1. download the Kaggle dataset      -> data/raw/Hotel_Reviews.csv
  2. clean + chronological split      -> data/processed/{train,val,test}.parquet
  3. BM25 indexes (train split)       -> data/bm25/<city>_<sentiment>/
"""
import argparse
import logging

from app.bm25_index import INDEX_DIR, build_index
from app.data_prep import PROCESSED_DIR, prepare
from app.dataset import RAW_DATA_DIR, download_dataset

CSV_PATH = RAW_DATA_DIR / "Hotel_Reviews.csv"


def main(rebuild: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if rebuild or not CSV_PATH.exists():
        print("Downloading dataset from Kaggle ...")
        download_dataset()
    if rebuild or not (PROCESSED_DIR / "train.parquet").exists():
        print("Cleaning and splitting ...")
        prepare(CSV_PATH)
    if rebuild or not (INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
        print("Building BM25 indexes ...")
        print(f"Indexed {build_index():,} chunks")
    print("Ready. Start the service with: uvicorn app.api:app")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="redo every step even if its output exists")
    main(parser.parse_args().rebuild)
