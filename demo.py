"""
Quick demonstration and verification script for YOLO Object Detection System.
Generates a sample image, runs inference, extracts colors, calculates distances, and exports analytics.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import YOLODetector
from core.utils import export_detections_csv, export_detections_json

def create_sample_image(filepath: str = "sample_data/sample.jpg"):
    """Create a sample synthetic test image if no image exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Create 640x640 canvas
    img = np.ones((640, 640, 3), dtype=np.uint8) * 240
    
    # Draw simple shapes representing target objects
    cv2.rectangle(img, (80, 150), (280, 450), (50, 50, 220), -1)   # Red object
    cv2.circle(img, (450, 250), 90, (220, 80, 50), -1)             # Blue object
    cv2.ellipse(img, (320, 500), (100, 60), 0, 0, 360, (50, 200, 50), -1) # Green object

    cv2.putText(img, "YOLO Color & Distance Test Scene", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(filepath, img)
    return filepath

def main():
    print("🎯 YOLO Object Detection, Color & Distance Analytics - Verification")
    sample_img_path = "sample_data/sample.jpg"
    create_sample_image(sample_img_path)
    
    print("\n[1/3] Initializing YOLO Detector Engine...")
    detector = YOLODetector(model_path="yolov8n.pt", conf_threshold=0.20)
    
    print("\n[2/3] Running Inference with Color Extraction & Distance Measurement...")
    output_annotated_path = "outputs/demo_annotated.jpg"
    res = detector.process_image(
        sample_img_path,
        output_path=output_annotated_path,
        detect_color=True,
        measure_distance=True,
        distance_threshold=200.0
    )

    print(f"\n✅ Detection Complete!")
    print(f"Total Objects Found: {res.total_count}")
    print(f"Counts per class: {res.counts_by_class}")
    
    for i, d in enumerate(res.detections):
        print(f"  - Object #{i+1}: {d['class_name']} | Color: {d.get('color_name', 'Unknown')} ({d.get('color_hex', '')}) | Conf: {d['confidence']}")
        
    if res.distances:
        print(f"\n📏 Inter-Object Distances:")
        for p in res.distances:
            alert_str = "🚨 ALERT (< 200px)" if p["is_close"] else "🟢 SAFE"
            print(f"  - {p['obj_a_label']} <---> {p['obj_b_label']}: {p['distance_px']} px [{alert_str}]")
        print(f"Closest Distance: {res.closest_distance} px")
    else:
        print("  - Distance: Single or no objects detected in synthetic test.")

    print(f"\nAnnotated Image Saved: {output_annotated_path}")

    print("\n[3/3] Exporting Analytics Data...")
    csv_file = export_detections_csv(res.detections, "outputs/demo_detections.csv")
    json_file = export_detections_json(res.detections, "outputs/demo_detections.json")
    print(f"CSV Report: {csv_file}")
    print(f"JSON Report: {json_file}")

if __name__ == "__main__":
    main()
