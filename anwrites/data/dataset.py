import os
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from .parser import parse_omniglot_txt
from .normalization import preprocess_sequence

log = logging.getLogger(__name__)

class OmniglotStrokeDataset(Dataset):
    """
    Loading data.
    """
    def __init__(self, data_dir: str, rdp_epsilon: float = 2.0):
        super().__init__()

        self.rdp_epsilon = rdp_epsilon
        self.file_paths = list(Path(data_dir).rglob("*.txt"))
        self.file_paths = [
            p for p in self.file_paths if "_MACOSX" not in str(p)
        ]

        if not self.file_paths:
            log.warning(f".txt file not found in {data_dir}")
        
        log.info(f"Dataset found: {len(self.file_paths)} sample.")

    def __len__(self) -> int:
        """Returns the total number of samples (files)."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves a single data sample based on the 'idx' index.
        """
        file_path = self.file_paths[idx]

        label, raw_strokes = parse_omniglot_txt(str(file_path))

        if not raw_strokes:
            log.warning(f"File is corrupted, loading another random sample: {file_path}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        
        sequence = preprocess_sequence(raw_strokes, self.rdp_epsilon)

        if sequence.size == 0:
            log.warning(f"Preprocessing failed, loading another random sample: {file_path}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)

        return sequence_tensor, label