import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ROIManager:
    """Manages Region of Interest (ROI) polygon zones and filtering."""
    def __init__(self, polygon_pts: Optional[List[List[float]]] = None):
        """
        polygon_pts: List of normalized points [[x, y], ...] in 0.0 - 1.0 range.
        """
        self.polygon_pts = polygon_pts or []

    def is_empty(self) -> bool:
        return len(self.polygon_pts) < 3

    def get_pixel_polygon(self, frame_width: int, frame_height: int) -> np.ndarray:
        if self.is_empty():
            return np.array([], dtype=np.int32)
        pts = np.array(
            [[int(pt[0] * frame_width), int(pt[1] * frame_height)] for pt in self.polygon_pts],
            dtype=np.int32
        )
        return pts

    def filter_detections(self, detections: List[Dict[str, Any]], frame_width: int, frame_height: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits detections into (inside_roi, outside_roi).
        Uses bottom-center of bounding box for ROI test (standard for ground contact).
        """
        if self.is_empty():
            return detections, []

        poly = self.get_pixel_polygon(frame_width, frame_height)
        inside = []
        outside = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Bottom center point
            cx = (x1 + x2) / 2.0
            cy = y2
            
            res = cv2.pointPolygonTest(poly, (cx, cy), False)
            if res >= 0:
                inside.append(det)
            else:
                outside.append(det)

        return inside, outside

    def draw_roi(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), alpha: float = 0.25) -> np.ndarray:
        """Draw filled translucent ROI polygon on image."""
        if self.is_empty():
            return image

        h, w = image.shape[:2]
        poly = self.get_pixel_polygon(w, h)
        
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Blend overlay
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image

class LineCounter:
    """Tracks objects crossing a virtual boundary line."""
    def __init__(self, line_pt1: List[float] = [0.1, 0.5], line_pt2: List[float] = [0.9, 0.5]):
        """
        line_pt1, line_pt2: Normalized coordinates [x, y] in range 0.0 - 1.0
        """
        self.line_pt1 = line_pt1
        self.line_pt2 = line_pt2
        self.counts_in = 0
        self.counts_out = 0
        self.track_positions: Dict[int, List[Tuple[float, float]]] = {}
        self.crossed_ids = set()

    def _cross_product(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Cross product to check which side of line segment (p1 -> p2) point p3 lies on."""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    def update(self, detections: List[Dict[str, Any]], frame_width: int, frame_height: int):
        """Update counter with current frame's detections containing track_ids."""
        p1 = (self.line_pt1[0] * frame_width, self.line_pt1[1] * frame_height)
        p2 = (self.line_pt2[0] * frame_width, self.line_pt2[1] * frame_height)

        current_tracks = set()
        for det in detections:
            track_id = det.get("track_id", None)
            if track_id is None:
                continue
            
            track_id = int(track_id)
            current_tracks.add(track_id)
            
            x1, y1, x2, y2 = det["bbox"]
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            if track_id not in self.track_positions:
                self.track_positions[track_id] = [center]
            else:
                self.track_positions[track_id].append(center)
                if len(self.track_positions[track_id]) > 5:
                    self.track_positions[track_id].pop(0)

            # Check line crossing if we have at least 2 consecutive positions
            if len(self.track_positions[track_id]) >= 2 and track_id not in self.crossed_ids:
                prev_pt = self.track_positions[track_id][-2]
                curr_pt = self.track_positions[track_id][-1]
                
                cp_prev = self._cross_product(p1, p2, prev_pt)
                cp_curr = self._cross_product(p1, p2, curr_pt)
                
                # If sign changes, line segment crossed
                if cp_prev * cp_curr < 0:
                    if cp_curr > 0:
                        self.counts_in += 1
                    else:
                        self.counts_out += 1
                    self.crossed_ids.add(track_id)

        # Cleanup old track positions
        dead_tracks = set(self.track_positions.keys()) - current_tracks
        for dt in dead_tracks:
            # Keep in crossed_ids to avoid re-counting, but prune position history
            if len(self.track_positions[dt]) > 0:
                self.track_positions.pop(dt, None)

    def draw_line(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Draw boundary line and counter HUD on image."""
        h, w = image.shape[:2]
        p1 = (int(self.line_pt1[0] * w), int(self.line_pt1[1] * h))
        p2 = (int(self.line_pt2[0] * w), int(self.line_pt2[1] * h))

        # Draw line
        cv2.line(image, p1, p2, color, 3, cv2.LINE_AA)
        
        # Draw end dots
        cv2.circle(image, p1, 6, (0, 255, 0), -1)
        cv2.circle(image, p2, 6, (0, 255, 0), -1)

        # Draw HUD text panel
        hud_text = f"IN: {self.counts_in} | OUT: {self.counts_out}"
        cv2.rectangle(image, (10, 10), (220, 50), (0, 0, 0), -1)
        cv2.putText(image, hud_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return image
