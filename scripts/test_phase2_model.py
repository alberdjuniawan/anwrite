import os
import sys
import logging
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from anwrites.data.dataset import OmniglotStrokeDataset
from anwrites.model.rnn import HandwritingRNN
from scripts.test_phase1_dataloader import collate_fn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    DATA_DIR = "data/strokes_background"
    
    logging.info("-Starting Phase 2 Test (Model Test)-")
    
    try:
        dataset = OmniglotStrokeDataset(data_dir=DATA_DIR)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        inputs, labels = next(iter(loader))
        logging.info(f"Batch ready. Shape: {inputs.shape}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    INPUT_DIM = 3     
    HIDDEN_DIM = 256  
    NUM_MIXTURES = 20 
    
    model = HandwritingRNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_mixtures=NUM_MIXTURES
    )
    
    logging.info(f"HandwritingRNN initialized (Hidden={HIDDEN_DIM}, Mixtures={NUM_MIXTURES})")

    try:
        logging.info("Running forward pass...")
        pi, sigma, mu, pen_up, _ = model(inputs)
        
        logging.info("Forward pass successful!")
        logging.info(f"  - pi shape: {pi.shape}")
        logging.info(f"  - sigma shape: {sigma.shape}")
        logging.info(f"  - mu shape: {mu.shape}")
        logging.info(f"  - pen_up shape: {pen_up.shape}")
        
        logging.info("Computing loss...")
        total_loss, mdn, p1 = model.calculate_loss(pi, sigma, mu, pen_up, inputs)
        
        logging.info("Loss computed successfully!")
        logging.info(f"  - Total Loss: {total_loss.item():.4f}")
        logging.info(f"  - MDN Loss: {mdn.item():.4f}")
        logging.info(f"  - PenUp Loss: {p1.item():.4f}")
        
        logging.info("-Model Test Successful!-")
        
    except Exception as e:
        logging.error(f"Model execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()