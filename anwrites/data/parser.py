import os
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

log = logging.getLogger(__name__)

StrokeList = List[List[Dict[str, int]]]
ParsedData = Tuple[Optional[str], Optional[StrokeList]]

def parse_omniglot_txt(file_path: str) -> ParsedData:
    """
    Reading a single .txt file from strokes dataset
    and extracting the sequence of strokes.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        processed_strokes: StrokeList = []
        current_stroke: List[Dict[str, int]] = []

        try:
            parts = Path(file_path).parts
            label = f"{parts[-3]}/{parts[-2]}"
        except Exception:
            label = "unknown"
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue

            if line == "START":
                continue

            if line == "BREAK":
                if current_stroke:
                    processed_strokes.append(current_stroke)
                current_stroke = []
            else:
                try:
                    x_str, y_str, t_str = line.split(',')
                    current_stroke.append({
                        'x': int(float(x_str)),
                        'y': int(float(y_str)),
                        't': int(float(t_str))
                    })
                except ValueError:
                    log.warning(f"Skipping malformed line in {file_path}: {line}")
        if current_stroke:
            processed_strokes.append(current_stroke)

        if not processed_strokes:
            log.warning(f"no valid strokes were extracted from {file_path}")
            return None, None
        
        return label, processed_strokes
    
    except Exception as e:
        log.error(f"Error reading .txt file {file_path}: {e}", exc_info=True)
        return None, None