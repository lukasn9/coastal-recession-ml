import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def encode_rle(binary_mask: np.ndarray) -> dict:
    """Encode a boolean (H, W) mask as standard COCO RLE (the format Roboflow imports as 'COCO RLE')."""
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype("uint8")))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def mask_area_and_bbox(binary_mask: np.ndarray) -> tuple[float, list[float]]:
    """Pixel area and COCO-style [x, y, width, height] bounding box of a boolean mask."""
    rows = np.where(np.any(binary_mask, axis=1))[0]
    cols = np.where(np.any(binary_mask, axis=0))[0]
    if rows.size == 0:
        return 0.0, [0.0, 0.0, 0.0, 0.0]
    y_min, y_max = rows[0], rows[-1]
    x_min, x_max = cols[0], cols[-1]
    area = float(binary_mask.sum())
    bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
    return area, bbox


def write_tile_image(output_dir: Path, split: str, tile_name: str, rgb_tile: np.ndarray) -> None:
    """Write one tile's RGB image into {output_dir}/{split}/{tile_name} (Roboflow's per-split COCO layout)."""
    image_path = output_dir / split / tile_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.transpose(rgb_tile, (1, 2, 0))).save(image_path)


class CocoDatasetWriter:
    """
    Accumulates one COCO 'image' entry per tile and one RLE-encoded 'annotation'
    per class present in that tile (a single annotation can cover several
    disconnected regions — unlike polygons, RLE doesn't need one shape per blob).
    write() emits a Roboflow-importable _annotations.coco.json.
    """

    def __init__(self, class_names: list[str]):
        self.categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
        self.images = []
        self.annotations = []
        self._next_image_id = 1
        self._next_ann_id = 1

    def add_tile(self, file_name: str, width: int, height: int, label_tile: np.ndarray, class_id_map: dict[int, int]) -> int:
        """Register one tile's image entry and its per-class RLE annotations. Returns the assigned image id."""
        image_id = self._next_image_id
        self._next_image_id += 1
        self.images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})

        for raw_label, class_id in class_id_map.items():
            binary_mask = label_tile == raw_label
            if not binary_mask.any():
                continue
            area, bbox = mask_area_and_bbox(binary_mask)
            self.annotations.append(
                {
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": encode_rle(binary_mask),
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 1,
                }
            )
            self._next_ann_id += 1
        return image_id

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"images": self.images, "annotations": self.annotations, "categories": self.categories}, f)
