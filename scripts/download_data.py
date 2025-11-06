import kaggle
import os
import logging
import tarfile
from pathlib import Path
import zipfile

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

KAGGLE_DATASET_SLUG = "teykaicong/iamondb-handwriting-dataset"
DESTINATION_DIR = Path("data/iam_ondb")
XML_ARCHIVE_NAME = "xml.tgz"
XML_EXTRACT_DIR = DESTINATION_DIR / "xml"

def main():
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("Authenticating with Kaggle API...")
    try:
        kaggle.api.authenticate()
        logging.info("Authentication successful.")
    except Exception as e:
        logging.error("Authentication failed. Make sure kaggle.json exists in C:/Users/ASUS/.kaggle/")
        logging.error(f"Error: {e}")
        return

    logging.info(f"Downloading dataset: {KAGGLE_DATASET_SLUG}...")
    try:
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET_SLUG,
            path=DESTINATION_DIR,
            unzip=False
        )
        logging.info(f"Dataset downloaded to: {DESTINATION_DIR}")
    except Exception as e:
        logging.error("Failed to download dataset. Check your Kaggle API key permissions.")
        logging.error(f"Error: {e}")
        return

    zip_path = DESTINATION_DIR / f"{KAGGLE_DATASET_SLUG.split('/')[1]}.zip"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            logging.info(f"Extracting archive: {zip_path}...")
            zip_ref.extractall(DESTINATION_DIR)
        logging.info("Archive extracted successfully.")
        os.remove(zip_path)
    except Exception as e:
        logging.error(f"Failed to extract .zip file: {e}")
        return

    xml_tgz_path = DESTINATION_DIR / XML_ARCHIVE_NAME
    if not xml_tgz_path.exists():
        logging.error(f"Critical file {XML_ARCHIVE_NAME} not found!")
        return
        
    logging.info(f"Found {XML_ARCHIVE_NAME}. Extracting XML stroke data...")
    XML_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(xml_tgz_path, "r:gz") as tar:
            tar.extractall(path=XML_EXTRACT_DIR)
        logging.info("="*50)
        logging.info(f"SUCCESS! XML stroke data is ready at:")
        logging.info(f"{XML_EXTRACT_DIR.resolve()}")
        logging.info("="*50)
    except Exception as e:
        logging.error(f"Failed to extract {xml_tgz_path}: {e}")

if __name__ == "__main__":
    main()
