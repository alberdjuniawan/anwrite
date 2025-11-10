import os
import sys
import logging
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from anwrites.data.parser import parse_omniglot_txt
from anwrites.data.normalization import preprocess_sequence

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    TEST_FILE_PATH = 'data/strokes_background/Balinese/character03/0110_05.txt'

    if not os.path.exists(TEST_FILE_PATH):
        logging.error(f"Test file not found: {TEST_FILE_PATH}")
        return
    
    logging.info("start Pipeline phase 1 test...")
    logging.info(f"Parsing file: {TEST_FILE_PATH}")

    label, raw_strokes = parse_omniglot_txt(TEST_FILE_PATH)

    if not raw_strokes:
        logging.error("Failed to parse the file. Stopping.")
        return
    
    logging.info(f"Label: {label}")
    logging.info(f"Raw Data: {len(raw_strokes)} strokes")

    print("-" * 30)

    logging.info("Running preprocessing (RDP + Delta Conversion)...")
    sequence = preprocess_sequence(raw_strokes, rdp_epsilon=2.0)

    logging.info("\n -Sequence Data Ready for Model-")
    logging.info(f"Array Shape: {sequence.shape}")

    logging.info("\nFirst 10 points [dx, dy, p1]:")
    print(sequence[:10])

    pen_ups = np.sum(sequence[:, 2])
    logging.info(f"\nTotal Strokes: {len(raw_strokes)} | Total Pen-Ups (p1=1): {pen_ups}")

    assert len(raw_strokes) == pen_ups

    logging.info("Pen-Up verification: Success!")
    logging.info("-Pipeline phase 1 test finished-")

if __name__ == "__main__":
    main()