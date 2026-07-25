import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from src.inference import next_run_dir, resolve_input, run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained YOLO26 semantic segmentation model on one image or a "
        "folder of images, writing a class-ID mask and a colorized visualization per image."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to a trained .pt checkpoint")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A single image file, a folder, or a .zip file (e.g. a Roboflow export, "
        "unpacked automatically). If the resolved folder contains a 'test' subfolder, "
        "images are read from there; otherwise every image directly in the folder is used.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Project directory. Each run creates a new inference_N subfolder inside it, "
        "with masks/ and overlays/ inside that, so repeated runs never overwrite each other.",
    )
    args = parser.parse_args()

    model = YOLO(str(args.model))
    source = resolve_input(args.input)
    n = 1 if isinstance(source, Path) else len(source)
    print(f"Running inference on {n} image(s) with {args.model}")

    run_dir = next_run_dir(args.output_dir)
    summaries = run_inference(model, source, run_dir)

    for s in summaries:
        stats = ", ".join(f"{k}: {v:.2f}%" for k, v in s.items() if k != "file")
        print(f"  {s['file']}: {stats}")

    print(f"\nDone. Wrote masks and visualizations to {run_dir}")


if __name__ == "__main__":
    main()
