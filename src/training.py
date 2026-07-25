from pathlib import Path

import torch
from ultralytics import YOLO


def detect_device() -> str:
    """Best available torch device for this machine: CUDA (Colab) > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    data_yaml: Path,
    model: str = "yolo26n-sem.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    seed: int = 42,
    project: str = "runs/train",
    name: str = "coastal-seg",
):
    """
    Train a YOLO26 semantic segmentation model on a dataset in the Ultralytics
    semantic-segmentation layout (data_yaml pointing at images/{train,val} +
    masks_dir). Identical code path locally (MPS/CPU) and on Colab (CUDA) — device
    is auto-detected unless explicitly overridden, so this is the single function
    both scripts/train.py and the Colab notebook call.
    """
    device = device or detect_device()
    print(f"Model:  {model}")
    print(f"Data:   {data_yaml}")
    print(f"Device: {device}")

    yolo_model = YOLO(model)
    return yolo_model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        project=project,
        name=name,
    )
