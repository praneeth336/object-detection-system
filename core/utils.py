import os
import json
import time
import yaml
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    if not os.path.exists(config_path):
        # Fallback path if loaded from subdirectories
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}

class FPSCounter:
    """Utility class to track processing frames per second."""
    def __init__(self, buffer_len: int = 30):
        self.buffer_len = buffer_len
        self.start_time = None
        self.frame_times = []
    
    def start(self):
        self.start_time = time.time()
        self.frame_times.clear()
        
    def update(self) -> float:
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
            return 0.0
        
        elapsed = current_time - self.start_time
        self.start_time = current_time
        self.frame_times.append(elapsed)
        
        if len(self.frame_times) > self.buffer_len:
            self.frame_times.pop(0)
            
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def get_color_map(num_colors: int = 256) -> List[Tuple[int, int, int]]:
    """Generate a reproducible color map for visualization."""
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return [tuple(int(c) for c in color) for color in colors]

COLOR_MAP = get_color_map(256)

def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_conf: bool = True,
    show_tracks: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """
    Annotate an OpenCV BGR image with bounding boxes, labels, confidence scores, and tracking IDs.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    for det in detections:
        box = det["bbox"] # [x1, y1, x2, y2]
        cls_id = det.get("class_id", 0)
        cls_name = det.get("class_name", f"class_{cls_id}")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id", None)
        
        # Determine color by track_id if available, otherwise by class_id
        color_idx = int(track_id) if track_id is not None else int(cls_id)
        color = COLOR_MAP[color_idx % len(COLOR_MAP)]
        
        x1, y1, x2, y2 = map(int, box)
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Construct label text
        label_parts = []
        if show_tracks and track_id is not None:
            label_parts.append(f"#{track_id}")
        if show_labels:
            label_parts.append(cls_name)
        if show_conf:
            label_parts.append(f"{conf:.2f}")
            
        label = " | ".join(label_parts)
        
        if label:
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, 1)
            
            # Draw label background box
            lbl_y1 = max(0, y1 - text_height - 6)
            lbl_y2 = y1
            cv2.rectangle(annotated, (x1, lbl_y1), (x1 + text_width + 6, lbl_y2), color, -1)
            
            # Text color contrast (white or black depending on color brightness)
            brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(text_height + 2, y1 - 4)),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA
            )
            
    return annotated

def export_detections_csv(detections: List[Dict[str, Any]], filepath: str):
    """Export detection record list to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    rows = []
    for d in detections:
        row = {
            "frame_index": d.get("frame_index", 0),
            "timestamp": d.get("timestamp", ""),
            "class_id": d.get("class_id", -1),
            "class_name": d.get("class_name", ""),
            "confidence": d.get("confidence", 0.0),
            "track_id": d.get("track_id", ""),
            "x1": d["bbox"][0] if "bbox" in d else 0,
            "y1": d["bbox"][1] if "bbox" in d else 0,
            "x2": d["bbox"][2] if "bbox" in d else 0,
            "y2": d["bbox"][3] if "bbox" in d else 0,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath

def export_detections_json(detections: List[Dict[str, Any]], filepath: str):
    """Export detection record list to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(detections, f, indent=2)
    return filepath
