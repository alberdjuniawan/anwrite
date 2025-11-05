import kagglehub
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

KAGGLE_DATASET_SLUG = "sainijithm/online-handwriting-dataset"
DESTINATION_DIR = "data/online-handwriting-json"

def main():
    logging.info(f"Downloading dataset: {KAGGLE_DATASET_SLUG} ...")

    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        logging.info(f"Dataset successfully downloaded to cache: {path}")

    except Exception as e:
        logging.error("Failed to download dataset. Please check your Kaggle API credentials (kaggle.json).")
        logging.error(f"Exception details: {e}")
        return

    source_path = Path(path)
    dest_path = Path(DESTINATION_DIR)

    try:
        if dest_path.exists():
            logging.warning(f"Removing existing destination folder: {dest_path}")
            shutil.rmtree(dest_path)
        
        shutil.copytree(source_path, dest_path)

        logging.info("=" * 50)
        logging.info(f"Dataset successfully copied to: {dest_path.resolve()}")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"Failed to copy files from cache to {dest_path}")
        logging.error(f"Exception details: {e}")

if __name__ == "__main__":
    main()