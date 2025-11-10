import numpy as np
from rdp import rdp
import logging
from typing import List, Dict

log = logging.getLogger(__name__)
StrokeList = List[List[Dict[str, int]]]

def simplify_strokes(strokes: StrokeList, epsilon: float = 1.0) -> StrokeList:
    simplified_strokes = []
    total_points_before = sum(len(s) for s in strokes)

    for stroke in strokes:
        points = np.array([[p['x'], p['y']] for p in stroke])
        
        if len(points) > 2:
            mask = rdp(points, epsilon=epsilon, return_mask=True)
            simplified_stroke = [point for(point, keep) in zip(stroke, mask) if keep]
            if simplified_stroke:
                simplified_strokes.append(simplified_stroke)
        elif len(points) > 0:
            simplified_strokes.append(stroke)
            
    total_points_after = sum(len(s) for s in simplified_strokes)
    log.debug(f"RDP simplified {total_points_before} points -> {total_points_after} points")
    return simplified_strokes

def convert_to_delta_format(strokes: StrokeList) -> np.ndarray:
    sequence_data = []
    current_x, current_y = 0, 0
    
    for stroke in strokes:
        if not stroke: continue
            
        first_point = stroke[0]
        delta_x = first_point['x'] - current_x
        delta_y = first_point['y'] - current_y
        sequence_data.append([delta_x, delta_y, 0])
        current_x, current_y = first_point['x'], first_point['y']
        
        for i in range(1, len(stroke)):
            point = stroke[i]
            delta_x = point['x'] - current_x
            delta_y = point['y'] - current_y
            sequence_data.append([delta_x, delta_y, 0])
            current_x, current_y = point['x'], point['y']
        
        if sequence_data:
            sequence_data[-1][2] = 1

        if not sequence_data:
            return np.array([], dtype=np.int32)

    return np.array(sequence_data, dtype=np.int32)

def preprocess_sequence(strokes: StrokeList, rdp_epsilon: float = 1.0) -> np.ndarray:
    log.info(f"Preprocessing... Raw strokes: {len(strokes)}")
    simplified = simplify_strokes(strokes, epsilon=rdp_epsilon)
    log.info(f"strokes after RDP: {len(simplified)}")
    sequence = convert_to_delta_format(simplified)

    if sequence.size == 0:
        log.warning("Preprocessing resulted in an empty sequence.")
        return np.empty((0, 3), dtype=np.int32)
    
    log.info(f"Total sequence points: {sequence.shape[0]}")
    return sequence