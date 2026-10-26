import argparse
import os
import sys
import subprocess
import cv2

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import YOLODetector
from core.utils import export_detections_csv, export_detections_json
from core.trainer import YOLOTrainer

def cmd_detect_image(args):
    print(f"[YOLO CLI] Loading model: {args.model}...")
    detector = YOLODetector(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)

    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(exts)]
    else:
        print(f"Error: Input path '{args.input}' not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    all_detections = []

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        out_img_path = os.path.join(args.output_dir, f"det_{fname}")
        res = detector.process_image(
            img_path,
            output_path=out_img_path,
            detect_color=args.detect_color,
            measure_distance=args.measure_distance,
            distance_threshold=args.dist_thresh
        )
        all_detections.extend(res.detections)
        dist_info = f" | Closest dist: {res.closest_distance}px" if res.closest_distance else ""
        print(f"✅ Processed {fname} -> Found {res.total_count} objects{dist_info} | Saved: {out_img_path}")

    if args.save_csv:
        csv_file = os.path.join(args.output_dir, "detections.csv")
        export_detections_csv(all_detections, csv_file)
        print(f"📄 Saved CSV report to: {csv_file}")

    if args.save_json:
        json_file = os.path.join(args.output_dir, "detections.json")
        export_detections_json(all_detections, json_file)
        print(f"📄 Saved JSON report to: {json_file}")

def cmd_detect_video(args):
    print(f"[YOLO CLI] Processing video: {args.input} with model {args.model}...")
    detector = YOLODetector(model_path=args.model, conf_threshold=args.conf, iou_threshold=args.iou)

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(args.output_dir, exist_ok=True)
    out_video_path = os.path.join(args.output_dir, "annotated_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    all_video_dets = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = detector.detect_frame(
                frame,
                frame_index=frame_idx,
                enable_tracking=True,
                detect_color=args.detect_color,
                measure_distance=args.measure_distance,
                distance_threshold=args.dist_thresh
            )
            if res.annotated_image is not None:
                out_writer.write(res.annotated_image)
            all_video_dets.extend(res.detections)
            if frame_idx % 30 == 0:
                closest_info = f" | Closest: {res.closest_distance}px" if res.closest_distance else ""
                print(f"Frame {frame_idx} | Objects: {res.total_count}{closest_info} | FPS: {res.fps}")
            frame_idx += 1
    finally:
        cap.release()
        out_writer.release()

    print(f"🎬 Video processing complete! Output saved to: {out_video_path}")
    if args.save_csv:
        export_detections_csv(all_video_dets, os.path.join(args.output_dir, "video_detections.csv"))

def cmd_train(args):
    print(f"[YOLO CLI] Starting custom model training...")
    trainer = YOLOTrainer(base_model=args.model)
    res = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch
    )
    print(f"🎉 Training finished successfully! Best weights: {res.get('best_weights')}")

def cmd_evaluate(args):
    from evaluate import evaluate_model
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_dir=args.output_dir
    )

def cmd_serve_ui(args):
    print(f"[YOLO CLI] Launching Streamlit Web App on port {args.port}...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(args.port), "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd)

def cmd_serve_api(args):
    print(f"[YOLO CLI] Launching FastAPI server on {args.host}:{args.port}...")
    import uvicorn
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=True)

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection, Color & Spatial Tracking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # detect-image
    p_img = subparsers.add_parser("detect-image", help="Detect objects in single image or directory")
    p_img.add_argument("--input", "-i", required=True, help="Input image file path or directory")
    p_img.add_argument("--output-dir", "-o", default="outputs", help="Output directory for annotated images")
    p_img.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path or name")
    p_img.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_img.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    p_img.add_argument("--no-color", dest="detect_color", action="store_false", help="Disable color recognition")
    p_img.add_argument("--no-distance", dest="measure_distance", action="store_false", help="Disable distance measurement")
    p_img.add_argument("--dist-thresh", type=float, default=150.0, help="Proximity alert threshold in pixels")
    p_img.add_argument("--save-csv", action="store_true", help="Export detections to CSV")
    p_img.add_argument("--save-json", action="store_true", help="Export detections to JSON")
    p_img.set_defaults(func=cmd_detect_image, detect_color=True, measure_distance=True)

    # detect-video
    p_vid = subparsers.add_parser("detect-video", help="Detect and track objects in video or webcam stream")
    p_vid.add_argument("--input", "-i", required=True, help="Input video file path or webcam index (e.g. 0)")
    p_vid.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    p_vid.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path or name")
    p_vid.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_vid.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    p_vid.add_argument("--no-color", dest="detect_color", action="store_false", help="Disable color recognition")
    p_vid.add_argument("--no-distance", dest="measure_distance", action="store_false", help="Disable distance measurement")
    p_vid.add_argument("--dist-thresh", type=float, default=150.0, help="Proximity alert threshold in pixels")
    p_vid.add_argument("--save-csv", action="store_true", help="Export detections to CSV")
    p_vid.set_defaults(func=cmd_detect_video, detect_color=True, measure_distance=True)

    # train
    p_trn = subparsers.add_parser("train", help="Train custom YOLO model on custom dataset")
    p_trn.add_argument("--data", "-d", required=True, help="Path to data.yaml dataset file")
    p_trn.add_argument("--model", "-m", default="yolov8n.pt", help="Base model to fine-tune")
    p_trn.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    p_trn.add_argument("--batch", type=int, default=16, help="Batch size")
    p_trn.add_argument("--imgsz", type=int, default=640, help="Image size")
    p_trn.set_defaults(func=cmd_train)

    # evaluate
    p_eva = subparsers.add_parser("evaluate", help="Benchmark model accuracy (mAP50, mAP50-95, precision, recall, latency)")
    p_eva.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model path or name")
    p_eva.add_argument("--data", "-d", default="coco8.yaml", help="Validation dataset yaml config")
    p_eva.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    p_eva.add_argument("--batch", type=int, default=16, help="Batch size")
    p_eva.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda:0')")
    p_eva.add_argument("--output-dir", "-o", default="outputs", help="Output directory for reports")
    p_eva.set_defaults(func=cmd_evaluate)

    # serve-ui
    p_ui = subparsers.add_parser("serve-ui", help="Start interactive Streamlit web dashboard")
    p_ui.add_argument("--port", type=int, default=8501, help="Port for Streamlit server")
    p_ui.set_defaults(func=cmd_serve_ui)

    # serve-api
    p_api = subparsers.add_parser("serve-api", help="Start FastAPI REST server")
    p_api.add_argument("--host", default="0.0.0.0", help="API host IP")
    p_api.add_argument("--port", type=int, default=8000, help="API port")
    p_api.set_defaults(func=cmd_serve_api)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
