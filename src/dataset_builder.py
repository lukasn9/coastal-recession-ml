import random
from pathlib import Path

import numpy as np

from src import autolabel, coco_export, raster_io, semantic_export, tiling

CLASS_NAMES = ["water", "vegetation", "bare"]
_RAW_LABEL_TO_CLASS_ID = {
    autolabel.LABEL_WATER: 0,
    autolabel.LABEL_VEGETATION: 1,
    autolabel.LABEL_BARE: 2,
}

EXPORT_FORMATS = ("ultralytics", "coco")


def split_scenes(scene_dirs: list[Path], val_fraction: float, seed: int) -> dict[Path, str]:
    """Assign each scene entirely to 'train' or 'val' — never split within a scene, to avoid tile leakage."""
    shuffled = list(scene_dirs)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val_set = set(shuffled[:n_val])
    return {scene: ("val" if scene in val_set else "train") for scene in scene_dirs}


def tile_is_usable(label_tile: np.ndarray, min_valid_fraction: float) -> bool:
    """A tile is usable if enough of it is clear of cloud/fill (per the auto-label's invalid marker)."""
    return (label_tile != autolabel.LABEL_INVALID).mean() >= min_valid_fraction


def build_tile_dataset(
    region_dir: Path,
    output_dir: Path,
    export_format: str = "ultralytics",
    tile_size: int = 640,
    val_fraction: float = 0.2,
    min_valid_fraction: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Build a segmentation tile dataset from every preprocessed scene under region_dir.
    Scenes are split whole into train/val so neighboring tiles from the same scene
    never leak across the split. label.tif's own class scheme (0=water, 1=vegetation,
    2=bare, 255=invalid/ignore) is used directly, no remapping needed.

    export_format:
      - "ultralytics": images/{split} + masks/{split} PNGs + data.yaml, for local
        `yolo semantic train`.
      - "coco": images per split + a Roboflow-importable _annotations.coco.json
        with RLE-encoded segmentation masks (one annotation per class per tile,
        since RLE can represent several disconnected regions in a single mask).
    """
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"export_format must be one of {EXPORT_FORMATS}, got {export_format!r}")

    scene_dirs = sorted(
        p for p in region_dir.iterdir()
        if p.is_dir() and (p / "processed" / "label.tif").exists()
    )
    split_by_scene = split_scenes(scene_dirs, val_fraction, seed)

    counts = {"train_tiles": 0, "val_tiles": 0, "skipped_tiles": 0, "scenes": len(scene_dirs)}
    coco_writers = {"train": coco_export.CocoDatasetWriter(CLASS_NAMES), "val": coco_export.CocoDatasetWriter(CLASS_NAMES)}

    for scene_dir in scene_dirs:
        split = split_by_scene[scene_dir]
        processed_dir = scene_dir / "processed"

        rgb, _ = raster_io.read_multiband(processed_dir / "rgb.tif")
        label, _ = raster_io.read_band(processed_dir / "label.tif")

        height, width = label.shape
        for bounds in tiling.tile_bounds(height, width, tile_size):
            label_tile = tiling.extract_tile(label, bounds)
            if not tile_is_usable(label_tile, min_valid_fraction):
                counts["skipped_tiles"] += 1
                continue

            rgb_tile = tiling.extract_tile(rgb, bounds)
            row_start, col_start, _, _ = bounds
            tile_name = f"{scene_dir.name}_r{row_start}_c{col_start}.png"

            if export_format == "ultralytics":
                semantic_export.write_tile(output_dir, split, tile_name, rgb_tile, label_tile)
            else:
                coco_export.write_tile_image(output_dir, split, tile_name, rgb_tile)
                coco_writers[split].add_tile(tile_name, tile_size, tile_size, label_tile, _RAW_LABEL_TO_CLASS_ID)

            counts[f"{split}_tiles"] += 1

    if export_format == "ultralytics":
        semantic_export.write_data_yaml(output_dir, CLASS_NAMES)
    else:
        for split, writer in coco_writers.items():
            writer.write(output_dir / split / "_annotations.coco.json")

    return counts
