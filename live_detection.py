"""
Real-time Live Webcam & Camera Stream Object Detection Application.
Displays interactive live feed with YOLO detection, dominant color classification,
inter-object distance measurement, and proximity alert overlays.

Keyboard Shortcuts:
  'q' - Quit application
  'c' - Toggle color detection on/off
  'd' - Toggle distance measurement on/off
  't' - Toggle multi-object tracking (MOT) on/off
  's' - Save snapshot of current annotated frame
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

# Ensure root directory is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import YOLODetector

def run_live_detection(
    source=0,
    model_path="yolov8n.pt",
    conf=0.25,
    iou=0.45,
    enable_color=True,
    enable_distance=True,
    dist_threshold=150.0,
    enable_tracking=True,
    output_dir="outputs"
):
    print("=" * 65)
    print("📹 Starting Real-time Live Object Detection")
    print(f"Camera Source : {source}")
    print(f"Model         : {model_path}")
    print(f"Confidence    : {conf} | IoU: {iou}")
    print("Hotkeys: [Q] Quit | [C] Color | [D] Distance | [T] Tracking | [S] Snapshot")
    print("=" * 65)

    # Initialize YOLO detector
    detector = YOLODetector(model_path=model_path, conf_threshold=conf, iou_threshold=iou)

    # Convert source to int if integer index
    cam_source = int(source) if str(source).isdigit() else source

    # Open video capture (try DirectShow on Windows if int, fallback to default)
    if isinstance(cam_source, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_source)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera source '{source}'.")
        print("💡 Hint: If using USB webcam, check device index (0, 1, 2) or ensure no other app is using it.")
        print("💡 If using an IP Webcam app (DroidCam, IP Webcam), pass its URL e.g. http://192.168.1.x:8080/video")
        return

    # Set preferred resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = f"YOLO Live Detection - [{model_path}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1020, 680)

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    snapshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Warning: Received empty frame from camera. Waiting...")
                time.sleep(0.1)
                continue

            # Run detection on live frame
            res = detector.detect_frame(
                frame=frame,
                frame_index=frame_idx,
                enable_tracking=enable_tracking,
                detect_color=enable_color,
                measure_distance=enable_distance,
                distance_threshold=dist_threshold,
                annotate=True
            )

            annotated = res.annotated_image if res.annotated_image is not None else frame.copy()
            h, w = annotated.shape[:2]

            # Draw HUD Info Bar (Top)
            cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
            hud_text = f"FPS: {res.fps} | Objects: {res.total_count}"
            if enable_distance and res.closest_distance is not None:
                hud_text += f" | Closest: {res.closest_distance}px"
            
            # Check proximity alerts
            close_count = sum(1 for p in res.distances if p.get("is_close", False))
            if close_count > 0:
                hud_text += f" | ! PROXIMITY ALERT ({close_count}) !"

            cv2.putText(
                annotated,
                hud_text,
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if close_count > 0 else (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # Draw HUD Status Controls (Bottom)
            cv2.rectangle(annotated, (0, h - 30), (w, h), (30, 30, 30), -1)
            c_status = "ON" if enable_color else "OFF"
            d_status = "ON" if enable_distance else "OFF"
            t_status = "ON" if enable_tracking else "OFF"
            ctrl_text = f"[C] Color: {c_status} | [D] Distance: {d_status} | [T] Track: {t_status} | [S] Snapshot | [Q] Quit"
            cv2.putText(annotated, ctrl_text, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated)
            frame_idx += 1

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                print("🛑 Stopping live detection...")
                break
            elif key == ord('c'):
                enable_color = not enable_color
                print(f"🎨 Color detection: {'ENABLED' if enable_color else 'DISABLED'}")
            elif key == ord('d'):
                enable_distance = not enable_distance
                print(f"📏 Distance measurement: {'ENABLED' if enable_distance else 'DISABLED'}")
            elif key == ord('t'):
                enable_tracking = not enable_tracking
                print(f"🎯 MOT Tracking: {'ENABLED' if enable_tracking else 'DISABLED'}")
            elif key == ord('s'):
                snapshot_count += 1
                snap_path = os.path.join(output_dir, f"live_snapshot_{int(time.time())}_{snapshot_count}.jpg")
                cv2.imwrite(snap_path, annotated)
                print(f"📸 Saved snapshot to: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released and window closed.")

def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO Live Camera Detection")
    parser.add_argument("--source", "-s", default="0", help="Camera index (0, 1) or IP stream URL (e.g. http://192.168.1.10:8080/video)")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path or name")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--no-color", dest="enable_color", action="store_false", help="Disable dominant color recognition")
    parser.add_argument("--no-distance", dest="enable_distance", action="store_false", help="Disable distance measurement")
    parser.add_argument("--dist-thresh", type=float, default=150.0, help="Proximity alert threshold in pixels")
    parser.add_argument("--no-tracking", dest="enable_tracking", action="store_false", help="Disable object tracking")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Directory to save snapshots")

    args = parser.parse_args()
    run_live_detection(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        enable_color=args.enable_color,
        enable_distance=args.enable_distance,
        dist_threshold=args.dist_thresh,
        enable_tracking=args.enable_tracking,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
