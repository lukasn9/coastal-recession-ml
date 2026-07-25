import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 semantic segmentation model. Works the same locally "
        "(MPS/CPU) or on Colab (CUDA) — device is auto-detected unless overridden. "
        "Point --data at a data.yaml from either `main.py build-dataset` (local) or "
        "`scripts/prepare_coco_dataset.py` (Roboflow COCO export)."
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to a semantic-segmentation data.yaml",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-sem.pt",
        help="Model checkpoint/config to start from (default: yolo26n-sem.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. '0' (CUDA), 'mps', 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--project", default="runs/train", help="Output project directory (default: runs/train)")
    parser.add_argument("--name", default="coastal-seg", help="Run name (default: coastal-seg)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"data.yaml not found: {args.data}")
        return

    train_model(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
