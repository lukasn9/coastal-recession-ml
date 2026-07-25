"""
Single entry point for the coastal-recession-ml pipeline.

Usage:
    python main.py <command> [args...]
    python main.py <command> --help    (for that command's own arguments)

Commands:
    download        Download Landsat scenes for a region       (scripts/dataset_download.py)
    preprocess      Cloud mask, NDWI/NDVI, RGB, auto-label      (scripts/preprocess.py)
    build-dataset   Build a YOLO tile dataset (local formats)   (scripts/build_yolo_dataset.py)
    prepare-coco    Convert a Roboflow/COCO export for training (scripts/prepare_coco_dataset.py)
    train           Train a YOLO26 semantic segmentation model  (scripts/train.py)
"""

import argparse
import importlib
import sys

COMMANDS = {
    "download": "scripts.dataset_download",
    "preprocess": "scripts.preprocess",
    "build-dataset": "scripts.build_yolo_dataset",
    "prepare-coco": "scripts.prepare_coco_dataset",
    "train": "scripts.train",
}


def main():
    parser = argparse.ArgumentParser(
        description="Coastal recession ML pipeline. Run `python main.py <command> --help` "
        "for a command's own arguments.",
    )
    parser.add_argument("command", choices=COMMANDS.keys(), help="Pipeline stage to run")
    parser.add_argument("command_args", nargs=argparse.REMAINDER, help="Arguments passed through to that stage")
    args = parser.parse_args()

    module = importlib.import_module(COMMANDS[args.command])
    sys.argv = [args.command] + args.command_args
    module.main()


if __name__ == "__main__":
    main()
