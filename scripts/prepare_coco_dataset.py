import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.archive_utils import extract_if_zip
from src.coco_to_masks import convert_coco_split
from src.run_dirs import next_run_dir
from src.semantic_export import write_data_yaml

# Roboflow exports validation images under "valid/"; our own coco_export.py uses "val/".
SPLIT_DIR_CANDIDATES = {
    "train": ["train"],
    "val": ["val", "valid", "validation"],
}


def find_split_dir(root: Path, candidates: list[str]) -> Path:
    """
    Find a split folder (e.g. train/) anywhere under root, not just as a direct
    child. Roboflow doesn't always put train/valid/test at the zip's top level,
    some export settings nest everything inside an extra project/version folder.
    """
    for name in candidates:
        if (root / name / "_annotations.coco.json").exists():
            return root / name
        for candidate in root.rglob(name):
            if candidate.is_dir() and (candidate / "_annotations.coco.json").exists():
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
        help="Root of the COCO dataset, or a .zip of one (unpacked automatically). "
        "Contains train/ and val(id)/ subfolders, each with _annotations.coco.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Project directory. Each run creates a new dataset_N subfolder inside it, "
        "so repeated runs never overwrite each other.",
    )
    args = parser.parse_args()

    input_dir = extract_if_zip(args.input_dir, prefix="coco_input_")
    train_dir = find_split_dir(input_dir, SPLIT_DIR_CANDIDATES["train"])
    val_dir = find_split_dir(input_dir, SPLIT_DIR_CANDIDATES["val"])

    run_dir = next_run_dir(args.output_dir, prefix="dataset")
    print(f"Input:  {args.input_dir}")
    print(f"  train split: {train_dir}")
    print(f"  val split:   {val_dir}")
    print(f"Output: {run_dir}")

    class_names = None
    for split, split_dir in [("train", train_dir), ("val", val_dir)]:
        names = convert_coco_split(
            coco_json_path=split_dir / "_annotations.coco.json",
            images_dir=split_dir,
            output_images_dir=run_dir / "images" / split,
            output_masks_dir=run_dir / "masks" / split,
        )
        print(f"  {split}: classes = {names}")
        if class_names is None:
            class_names = names
        elif class_names != names:
            print(
                f"  WARNING: class list differs between splits (train={class_names}, {split}={names}), "
                "using the train split's class list/order for data.yaml"
            )

    write_data_yaml(run_dir, class_names)
    print(f"\nDone. Wrote {run_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
