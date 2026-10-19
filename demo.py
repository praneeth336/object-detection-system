"""
Quick demonstration and verification script for YOLO Object Detection System.
Generates a sample image, runs inference, exports analytics, and displays detected object metadata.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import YOLODetector
from core.utils import export_detections_csv, export_detections_json
from core.roi import ROIManager, LineCounter

def create_sample_image(filepath: str = "sample_data/sample.jpg"):
    """Create a sample synthetic test image if no image exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Create 640x640 canvas
    img = np.ones((640, 640, 3), dtype=np.uint8) * 240
    
    # Draw simple shapes representing target objects
    cv2.rectangle(img, (100, 100), (300, 400), (50, 50, 200), -1) # Red rectangle
    cv2.circle(img, (450, 200), 80, (200, 50, 50), -1)           # Blue circle
    cv2.ellipse(img, (350, 480), (120, 60), 0, 0, 360, (50, 200, 50), -1) # Green oval

    # Text header
    cv2.putText(img, "YOLO Test Image Context", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(filepath, img)
    return filepath

def main():
    print("🎯 YOLO Object Detection System - Demo & Verification")
    sample_img_path = "sample_data/sample.jpg"
    create_sample_image(sample_img_path)
    
    print("\n[1/3] Initializing YOLOv8 Detector Engine...")
    detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.25)
    
    print("\n[2/3] Running Inference on Sample Image...")
    output_annotated_path = "outputs/demo_annotated.jpg"
    res = detector.process_image(sample_img_path, output_path=output_annotated_path)

    print(f"\n✅ Detection Complete!")
    print(f"Total Objects Found: {res.total_count}")
    print(f"Counts per class: {res.counts_by_class}")
    print(f"Annotated Image Saved: {output_annotated_path}")

    print("\n[3/3] Exporting Analytics Data...")
    csv_file = export_detections_csv(res.detections, "outputs/demo_detections.csv")
    json_file = export_detections_json(res.detections, "outputs/demo_detections.json")
    print(f"CSV Report: {csv_file}")
    print(f"JSON Report: {json_file}")

if __name__ == "__main__":
    main()
