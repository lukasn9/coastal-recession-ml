from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def write_tile(output_dir: Path, split: str, tile_name: str, rgb_tile: np.ndarray, label_tile: np.ndarray) -> None:
    """Write one tile's RGB image and its single-channel class-ID mask into the Ultralytics -sem layout."""
    image_path = output_dir / "images" / split / tile_name
    mask_path = output_dir / "masks" / split / tile_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.transpose(rgb_tile, (1, 2, 0))).save(image_path)
    Image.fromarray(label_tile, mode="L").save(mask_path)


def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> None:
    """
    Write the data.yaml for an Ultralytics YOLO semantic-segmentation dataset
    (images/{train,val} + masks/{train,val}, matching filenames across the two).
    """
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(dataset_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
