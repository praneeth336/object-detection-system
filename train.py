"""
Standalone custom YOLO model fine-tuning script.
Usage:
    python train.py --data sample_data/data.yaml --epochs 25 --batch 16
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trainer import YOLOTrainer

def main():
    parser = argparse.ArgumentParser(description="YOLO Model Custom Dataset Fine-Tuning")
    parser.add_argument("--data", "-d", default="sample_data/data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="Pretrained base model (e.g., yolov8n.pt, yolov8s.pt, yolo11n.pt)")
    parser.add_argument("--epochs", "-e", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--device", default="auto", help="Compute device ('cpu', 'cuda:0', or 'auto')")

    args = parser.parse_args()

    print(f"==================================================")
    print(f"🚀 Starting YOLO Custom Training")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.imgsz}")
    print(f"==================================================")

    trainer = YOLOTrainer(base_model=args.model)
    results = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )

    print("\n✅ Training Process Finished!")
    print(f"Saved weights directory: {results.get('save_dir')}")
    print(f"Best model weights: {results.get('best_weights')}")

if __name__ == "__main__":
    main()
