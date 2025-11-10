import os
import sys
import logging
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from anwrites.data.dataset import OmniglotStrokeDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    from torch.nn.utils.rnn import pad_sequence
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, labels

def main():
    DATA_DIR = "data/strokes_background"
    
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data folder not found: {DATA_DIR}")
        return
        
    logging.info("-Starting Test Phase (Dataset & DataLoader)-")
    
    logging.info(f"Loading dataset from: {DATA_DIR}...")
    dataset = OmniglotStrokeDataset(data_dir=DATA_DIR)
    
    if len(dataset) == 0:
        logging.error("Dataset is empty! Stopping.")
        return
        
    logging.info(f"Total samples found: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    logging.info("DataLoader created. Fetching 1 batch...")
    
    try:
        first_batch_sequences, first_batch_labels = next(iter(loader))
        
        logging.info("-DataLoader Test Successful!-")
        logging.info(f"Labels in first batch: {first_batch_labels}")
        logging.info(f"Batch tensor shape: {first_batch_sequences.shape}")
        logging.info(f"(Shape is [BatchSize, MaxSequenceLength, 3])")
        
    except Exception as e:
        logging.error(f"Failed to fetch data from DataLoader: {e}")
        logging.error("This often happens on Windows. Try setting num_workers=0 in DataLoader.")

if __name__ == "__main__":
    main()