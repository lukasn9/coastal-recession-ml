import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coasttrain_convert import build_coco_dataset
from src.run_dirs import next_run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Coast Train Landsat-8 export into a Roboflow-importable COCO "
        "dataset with a binary sediment/not_sediment class, split by site."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the Coast Train zip (e.g. Landsat8_11_001.zip) or an already-extracted folder",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Project directory. Each run creates a new dataset_N subfolder inside it, "
        "so repeated runs never overwrite each other.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of sites (not images) held out for validation (default: 0.2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the site-level train/val split (default: 42)")
    args = parser.parse_args()

    run_dir = next_run_dir(args.output_dir, prefix="dataset")
    print(f"Input:  {args.input}")
    print(f"Output: {run_dir}")

    counts = build_coco_dataset(args.input, run_dir, val_fraction=args.val_fraction, seed=args.seed)

    print(f"\nDone. {counts['sites']} sites found.")
    print(f"  train images: {counts['train_images']}")
    print(f"  val images:   {counts['val_images']}")


if __name__ == "__main__":
    main()
