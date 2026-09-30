"""
Single entry point for the coastal-recession-ml pipeline.

Usage:
    python main.py <command> [args...]
    python main.py <command> --help    (for that command's own arguments)

Commands:
    download        Download satellite scenes for a region      (scripts/dataset_download.py)
    preprocess      Cloud mask, NDWI/NDVI, RGB, auto-label      (scripts/preprocess.py)
    refine-sand     Split bare into sand/bare via a trained model (scripts/refine_sand.py)
    build-dataset   Build a YOLO tile dataset (local formats)   (scripts/build_yolo_dataset.py)
    prepare-coco    Convert a Roboflow/COCO export for training (scripts/prepare_coco_dataset.py)
    prepare-coasttrain  Convert Coast Train to a COCO dataset    (scripts/prepare_coasttrain_dataset.py)
    train           Train a YOLO26 semantic segmentation model  (scripts/train.py)
    predict         Run a trained model on image(s)             (scripts/predict.py)
    roboflow-config Set stored Roboflow credentials             (scripts/roboflow_config.py)
    upload-roboflow Upload a COCO dataset to a Roboflow project  (scripts/upload_roboflow.py)
"""

import argparse
import importlib
import sys

COMMANDS = {
    "download": "scripts.dataset_download",
    "preprocess": "scripts.preprocess",
    "refine-sand": "scripts.refine_sand",
    "build-dataset": "scripts.build_yolo_dataset",
    "prepare-coco": "scripts.prepare_coco_dataset",
    "prepare-coasttrain": "scripts.prepare_coasttrain_dataset",
    "train": "scripts.train",
    "predict": "scripts.predict",
    "roboflow-config": "scripts.roboflow_config",
    "upload-roboflow": "scripts.upload_roboflow",
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
