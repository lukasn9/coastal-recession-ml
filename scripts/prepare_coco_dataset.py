import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.coco_to_masks import convert_coco_split
from src.semantic_export import write_data_yaml

# Roboflow exports validation images under "valid/"; our own coco_export.py uses "val/".
SPLIT_DIR_CANDIDATES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
}


def find_split_dir(root: Path, candidates: list[str]) -> Path:
    for name in candidates:
        candidate = root / name
        if (candidate / "_annotations.coco.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a split folder with _annotations.coco.json under {root} (tried: {candidates})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a COCO-format segmentation dataset (our own coco_export.py "
        "output, or a Roboflow COCO export) into the Ultralytics semantic-segmentation "
        "layout (images/masks PNGs + data.yaml) that scripts/train.py trains on."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Root of the COCO dataset — contains train/ and val(id)/ subfolders, "
        "each with _annotations.coco.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write the converted images/masks/data.yaml",
    )
    args = parser.parse_args()

    train_dir = find_split_dir(args.input_dir, SPLIT_DIR_CANDIDATES["train"])
    val_dir = find_split_dir(args.input_dir, SPLIT_DIR_CANDIDATES["val"])

    print(f"Input:  {args.input_dir}")
    print(f"  train split: {train_dir}")
    print(f"  val split:   {val_dir}")
    print(f"Output: {args.output_dir}")

    class_names = None
    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        names = convert_coco_split(
            coco_json_path=split_dir / "_annotations.coco.json",
            images_dir=split_dir,
            output_images_dir=args.output_dir / "images" / split,
            output_masks_dir=args.output_dir / "masks" / split,
        )
        print(f"  {split}: classes = {names}")
        if class_names is None:
            class_names = names
        elif class_names != names:
            print(
                f"  WARNING: class list differs between splits (train={class_names}, {split}={names}) "
                "— using the train split's class list/order for data.yaml"
            )

    write_data_yaml(args.output_dir, class_names)
    print(f"\nDone. Wrote {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
