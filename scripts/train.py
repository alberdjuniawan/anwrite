import os
import sys
import logging
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from anwrites.data.dataset import OmniglotStrokeDataset
from anwrites.model.rnn import HandwritingRNN
from scripts.test_phase1_dataloader import collate_fn

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DATA_DIR = "data/strokes_background"
MODEL_SAVE_PATH = "models/"
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 0

INPUT_DIM = 3
HIDDEN_DIM = 512
NUM_MIXTURES = 20
NUM_LAYERS = 3

def main():
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info("Loading dataset...")
    dataset = OmniglotStrokeDataset(data_dir=DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    logging.info(f"Loading dataset: {len(dataset)} sample.")

    model = HandwritingRNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_mixtures=NUM_MIXTURES,
        num_layers=NUM_LAYERS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info(f"Model created. Ready to start {NUM_EPOCHS} epochs...")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(NUM_EPOCHS):
        logging.info(f"-Epoch {epoch+1}/{NUM_EPOCHS}-")
        
        model.train()
        total_epoch_loss = 0

        for i, (data_batch, _) in enumerate(loader):
            data_batch = data_batch.to(device)
            input_sequence = data_batch[:, :-1]
            target_sequence = data_batch[:, 1:]
            optimizer.zero_grad()
            pi, sigma, mu, pen_up, _ = model(input_sequence)
            loss, mdn_l, p1_l = model.calculate_loss(pi, sigma, mu, pen_up, target_sequence)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

            if(i + 1) % 50 == 0:
                logging.info(f"  Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f} (mdn: {mdn_l.item():.4f}, P1: {p1_l.item():.4f})")

        avg_loss = total_epoch_loss / len(loader)
        logging.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        save_file = f"{MODEL_SAVE_PATH}anwrites_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_file)
        logging.info(f"Model saved to: {save_file}")

    logging.info("-Training Complete!")

if __name__ == "__main__":
    main()