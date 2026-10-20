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

def classify_hsv_color(h: float, s: float, v: float) -> str:
    """Classify HSV values (H: 0-179, S: 0-255, V: 0-255) into readable color names."""
    if v < 45:
        return "Black"
    if s < 38 and v > 195:
        return "White"
    if s < 45:
        return "Gray"
    
    # Brown: Warm hue, moderate value and saturation
    if 8 <= h <= 25 and 45 <= v <= 150 and s > 60:
        return "Brown"
    
    # Standard hue ranges
    if h < 8 or h >= 170:
        return "Red"
    elif 8 <= h < 22:
        return "Orange"
    elif 22 <= h < 38:
        return "Yellow"
    elif 38 <= h < 85:
        return "Green"
    elif 85 <= h < 105:
        return "Cyan"
    elif 105 <= h < 135:
        return "Blue"
    elif 135 <= h < 155:
        return "Purple"
    elif 155 <= h < 170:
        return "Pink"
    return "Unknown"

def detect_dominant_color(image: np.ndarray, bbox: List[float]) -> Dict[str, Any]:
    """
    Extract the dominant color of the object bounded by bbox [x1, y1, x2, y2].
    Samples the inner region of the bounding box to minimize background noise.
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return {"name": "Unknown", "rgb": [128, 128, 128], "hex": "#808080"}
    
    # Sample inner 60% of bounding box
    cx1 = int(x1 + bw * 0.2)
    cx2 = int(x2 - bw * 0.2)
    cy1 = int(y1 + bh * 0.2)
    cy2 = int(y2 - bh * 0.2)
    
    if cx2 <= cx1 or cy2 <= cy1:
        crop = image[y1:y2, x1:x2]
    else:
        crop = image[cy1:cy2, cx1:cx2]
        
    if crop.size == 0:
        return {"name": "Unknown", "rgb": [128, 128, 128], "hex": "#808080"}
        
    # Resize crop to speed up clustering
    small_crop = cv2.resize(crop, (30, 30), interpolation=cv2.INTER_AREA)
    pixels_bgr = small_crop.reshape(-1, 3).astype(np.float32)
    
    # Use K-Means (k=3) to extract dominant color cluster
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    try:
        _, labels, centers = cv2.kmeans(pixels_bgr, 3, None, criteria, 3, flags)
        counts = np.bincount(labels.flatten())
        dom_bgr = centers[np.argmax(counts)].astype(int)
    except Exception:
        dom_bgr = np.median(pixels_bgr, axis=0).astype(int)
        
    b, g, r = int(dom_bgr[0]), int(dom_bgr[1]), int(dom_bgr[2])
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    
    # Convert BGR to HSV
    one_pixel = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(one_pixel, cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = float(hsv[0]), float(hsv[1]), float(hsv[2])
    
    color_name = classify_hsv_color(hue, sat, val)
    return {
        "name": color_name,
        "rgb": [r, g, b],
        "hex": hex_code
    }

def calculate_pairwise_distances(
    detections: List[Dict[str, Any]],
    alert_threshold: float = 150.0
) -> List[Dict[str, Any]]:
    """
    Computes Euclidean distances between all pairs of detected objects.
    """
    pairs = []
    n = len(detections)
    if n < 2:
        return pairs
        
    for i in range(n):
        det_a = detections[i]
        box_a = det_a["bbox"]
        c_a = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
        name_a = det_a.get("class_name", f"obj_{i}")
        id_a = det_a.get("track_id", i + 1)
        col_a = det_a.get("color_name", "")
        
        for j in range(i + 1, n):
            det_b = detections[j]
            box_b = det_b["bbox"]
            c_b = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
            name_b = det_b.get("class_name", f"obj_{j}")
            id_b = det_b.get("track_id", j + 1)
            col_b = det_b.get("color_name", "")
            
            dist_px = float(np.sqrt((c_a[0] - c_b[0]) ** 2 + (c_a[1] - c_b[1]) ** 2))
            
            if dist_px < alert_threshold:
                status = "Alert (Very Close)"
            elif dist_px < alert_threshold * 1.5:
                status = "Caution (Moderate)"
            else:
                status = "Safe (Separated)"
                
            pairs.append({
                "obj_a_idx": i,
                "obj_a_label": f"#{id_a} {name_a}" + (f" ({col_a})" if col_a else ""),
                "obj_b_idx": j,
                "obj_b_label": f"#{id_b} {name_b}" + (f" ({col_b})" if col_b else ""),
                "center_a": (int(c_a[0]), int(c_a[1])),
                "center_b": (int(c_b[0]), int(c_b[1])),
                "distance_px": round(dist_px, 1),
                "is_close": dist_px < alert_threshold,
                "status": status
            })
            
    # Sort pairs by distance ascending (closest first)
    pairs.sort(key=lambda p: p["distance_px"])
    return pairs

def draw_distance_lines(
    image: np.ndarray,
    pairs: List[Dict[str, Any]],
    alert_threshold: float = 150.0,
    connect_mode: str = "all", # "all", "close_only", "nearest_only"
    max_lines: int = 20
) -> np.ndarray:
    """
    Draw distance measurement lines, object center dots, and distance badges between objects.
    """
    annotated = image.copy()
    if not pairs:
        return annotated
        
    drawn = 0
    seen_nearest = set()
    
    for p in pairs:
        if drawn >= max_lines:
            break
            
        is_close = p["is_close"]
        if connect_mode == "close_only" and not is_close:
            continue
            
        if connect_mode == "nearest_only":
            idx_a = p["obj_a_idx"]
            idx_b = p["obj_b_idx"]
            if idx_a in seen_nearest and idx_b in seen_nearest:
                continue
            seen_nearest.add(idx_a)
            seen_nearest.add(idx_b)
            
        pt_a = p["center_a"]
        pt_b = p["center_b"]
        dist = p["distance_px"]
        
        # Color coding: Red if close alert, Yellow/Orange if moderate, Green if safe
        if dist < alert_threshold:
            line_color = (0, 0, 255) # Red (BGR)
            text_bg = (0, 0, 200)
        elif dist < alert_threshold * 1.5:
            line_color = (0, 215, 255) # Yellow/Orange
            text_bg = (0, 180, 200)
        else:
            line_color = (0, 255, 0) # Green
            text_bg = (0, 160, 0)
            
        # Draw center dots
        cv2.circle(annotated, pt_a, 5, line_color, -1)
        cv2.circle(annotated, pt_b, 5, line_color, -1)
        
        # Draw connecting line
        cv2.line(annotated, pt_a, pt_b, line_color, 2, cv2.LINE_AA)
        
        # Midpoint for label
        mid_x = (pt_a[0] + pt_b[0]) // 2
        mid_y = (pt_a[1] + pt_b[1]) // 2
        
        label = f"{int(dist)}px"
        if is_close:
            label = f"! {int(dist)}px"
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        
        # Badge background
        cv2.rectangle(
            annotated,
            (mid_x - tw // 2 - 4, mid_y - th // 2 - 4),
            (mid_x + tw // 2 + 4, mid_y + th // 2 + 4),
            text_bg,
            -1
        )
        cv2.rectangle(
            annotated,
            (mid_x - tw // 2 - 4, mid_y - th // 2 - 4),
            (mid_x + tw // 2 + 4, mid_y + th // 2 + 4),
            (255, 255, 255),
            1
        )
        cv2.putText(
            annotated,
            label,
            (mid_x - tw // 2, mid_y + th // 2 - 1),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        drawn += 1
        
    return annotated

def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_conf: bool = True,
    show_tracks: bool = True,
    show_color: bool = True,
    thickness: int = 2
) -> np.ndarray:
    """
    Annotate an OpenCV BGR image with bounding boxes, labels, confidence scores, color tags, and tracking IDs.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    
    for det in detections:
        box = det["bbox"] # [x1, y1, x2, y2]
        cls_id = det.get("class_id", 0)
        cls_name = det.get("class_name", f"class_{cls_id}")
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id", None)
        color_name = det.get("color_name", "")
        
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
            if show_color and color_name and color_name != "Unknown":
                label_parts.append(f"{cls_name} ({color_name})")
            else:
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
            
            # Text color contrast
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
            "color_name": d.get("color_name", ""),
            "color_hex": d.get("color_hex", ""),
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
