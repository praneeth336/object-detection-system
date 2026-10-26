"""
Accuracy and Performance Evaluation Script for YOLO Object Detection Models.
Computes mAP50, mAP50-95, Precision, Recall, F1-Score, Latency, and Per-Class Metrics.
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path

# Ensure local directories for YOLO configs, datasets, and runs
workspace_dir = Path(__file__).parent.resolve()
config_dir = workspace_dir / ".config"
os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
os.environ["MPLCONFIGDIR"] = str(workspace_dir / ".config" / "matplotlib")

from ultralytics import YOLO

def evaluate_model(
    model_path: str = "yolov8n.pt",
    data_yaml: str = "coco8.yaml",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu",
    output_dir: str = "outputs"
):
    print("=" * 65)
    print(f"📊 Running YOLO Accuracy & Performance Evaluation")
    print(f"Model: {model_path}")
    print(f"Validation Dataset: {data_yaml}")
    print(f"Image Resolution: {imgsz}x{imgsz} | Batch Size: {batch_size} | Device: {device}")
    print("=" * 65)

    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(workspace_dir / "runs"),
        name="evaluation",
        exist_ok=True,
        verbose=True
    )

    # Extract overall metrics
    map50_95 = float(metrics.box.map)
    map50 = float(metrics.box.map50)
    map75 = float(metrics.box.map75)
    mean_precision = float(metrics.box.mp)
    mean_recall = float(metrics.box.mr)
    f1_score = float((2 * mean_precision * mean_recall) / (mean_precision + mean_recall)) if (mean_precision + mean_recall) > 0 else 0.0

    speed = metrics.speed
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_latency_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = round(1000.0 / total_latency_ms, 1) if total_latency_ms > 0 else 0.0

    # Extract per-class metrics
    class_metrics = []
    class_names = metrics.names
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        c_name = class_names[cls_idx]
        res = metrics.box.class_result(i)
        p = float(res[0])
        r = float(res[1])
        ap50 = float(res[2])
        ap50_95 = float(res[3])
        f1 = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0

        class_metrics.append({
            "Class ID": int(cls_idx),
            "Class Name": c_name,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f1, 4),
            "mAP@50": round(ap50, 4),
            "mAP@50-95": round(ap50_95, 4)
        })

    # Summary report
    summary = {
        "model": model_path,
        "dataset": data_yaml,
        "overall_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "mAP75": round(map75, 4),
            "precision": round(mean_precision, 4),
            "recall": round(mean_recall, 4),
            "f1_score": round(f1_score, 4)
        },
        "speed_benchmark_ms": {
            "preprocess": round(preprocess_ms, 2),
            "inference": round(inference_ms, 2),
            "postprocess": round(postprocess_ms, 2),
            "total_latency_per_image": round(total_latency_ms, 2),
            "estimated_fps": fps
        },
        "per_class_breakdown": class_metrics,
        "artifact_dir": str(metrics.save_dir)
    }

    # Print Formatted Results Table
    print("\n" + "=" * 65)
    print("🎯 OVERALL ACCURACY BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  • Mean Average Precision (mAP@50)     : {map50 * 100:.2f}%")
    print(f"  • Strict Average Precision (mAP@50-95): {map50_95 * 100:.2f}%")
    print(f"  • Precision (P)                       : {mean_precision * 100:.2f}%")
    print(f"  • Recall (R)                          : {mean_recall * 100:.2f}%")
    print(f"  • Balanced F1-Score                   : {f1_score * 100:.2f}%")
    print(f"  • Pre-process Latency                 : {preprocess_ms:.2f} ms")
    print(f"  • Inference Latency                   : {inference_ms:.2f} ms")
    print(f"  • Post-process Latency                : {postprocess_ms:.2f} ms")
    print(f"  • Effective Processing Speed          : {fps} FPS")
    print("=" * 65)

    if class_metrics:
        print("\n📋 PER-CLASS ACCURACY BREAKDOWN:")
        df_classes = pd.DataFrame(class_metrics)
        print(df_classes.to_string(index=False))

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "model_accuracy_report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, "model_accuracy_per_class.csv")
    if class_metrics:
        df_classes.to_csv(csv_path, index=False)

    print(f"\n💾 Saved full JSON report to : {json_path}")
    print(f"💾 Saved class CSV report to : {csv_path}")
    print(f"📈 Visual evaluation curves in : {metrics.save_dir}")
    print("=" * 65)

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Model Accuracy and Latency")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--data", "-d", default="coco8.yaml", help="Dataset yaml config")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Compute device ('cpu' or 'cuda:0')")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory for reports")

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
