import os
from typing import Dict, Any, Optional
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class YOLOTrainer:
    """Wrapper for custom YOLO model training, validation, and export."""

    def __init__(self, base_model: str = "yolov8n.pt"):
        if YOLO is None:
            raise ImportError("Ultralytics library is not installed.")
        self.base_model = base_model
        self.model = YOLO(base_model)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch_size: int = 16,
        device: str = "auto",
        project: str = "runs/train",
        name: str = "exp",
        optimizer: str = "auto"
    ) -> Dict[str, Any]:
        """
        Train YOLO model on custom dataset configuration.
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset YAML file not found: {data_yaml}")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            optimizer=optimizer
        )
        return {
            "save_dir": str(self.model.trainer.save_dir),
            "best_weights": os.path.join(str(self.model.trainer.save_dir), "weights", "best.pt"),
            "last_weights": os.path.join(str(self.model.trainer.save_dir), "weights", "last.pt"),
            "metrics": self.model.trainer.metrics if hasattr(self.model.trainer, "metrics") else {}
        }

    def validate(self, data_yaml: str) -> Dict[str, Any]:
        """Validate model accuracy on dataset split."""
        metrics = self.model.val(data=data_yaml)
        return metrics.results_dict if hasattr(metrics, "results_dict") else {}

    def export_model(self, format: str = "onnx") -> str:
        """Export trained model to ONNX, Engine, TorchScript, or TFLite format."""
        exported_path = self.model.export(format=format)
        return str(exported_path)
