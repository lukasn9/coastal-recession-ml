import random
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import coco_export
from src.archive_utils import extract_if_zip

CLASS_NAMES = ["not_sediment", "sediment"]

# Coast Train's own classes never legitimately relevant here: cloud cover and
# missing data aren't "not sediment", they're just unusable pixels.
_IGNORED_CLASSES = {"cloud", "nodata", "unusual", "unknown"}
IGNORE_VALUE = 255

_SITE_PATTERN = re.compile(r"^([a-zA-Z]+)_\d{4}-")


def find_npz_files(input_path: Path) -> list[Path]:
    """Resolve a Coast Train zip or already-extracted folder into its .npz files."""
    input_path = extract_if_zip(input_path, prefix="coasttrain_")
    npz_files = sorted(input_path.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {input_path}")
    return npz_files


def site_from_filename(npz_path: Path) -> str:
    """Extract the site name from a Coast Train filename, e.g. 'duck' from 'duck_2014-01-04-...npz'."""
    match = _SITE_PATTERN.match(npz_path.name)
    return match.group(1) if match else npz_path.stem


def load_binary_sediment_label(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one Coast Train sample and reduce it to (image, binary_label): image is
    the (H, W, 3) uint8 RGB array; binary_label is (H, W) uint8 with 1 = sediment
    (sand), 0 = confidently something else, and IGNORE_VALUE for cloud/nodata/
    other unusable classes. The sediment channel index is read from each file's
    own 'classes' array rather than assumed, in case ordering ever differs.
    """
    data = np.load(npz_path, allow_pickle=True)
    image = data["image"]
    label = data["label"]
    classes = [str(c) for c in data["classes"]]

    label_ids = label.argmax(axis=-1)

    binary = np.zeros(label_ids.shape, dtype="uint8")
    for i, name in enumerate(classes[: label.shape[-1]]):
        if name == "sediment":
            binary[label_ids == i] = 1
        elif name in _IGNORED_CLASSES:
            binary[label_ids == i] = IGNORE_VALUE

    return image, binary


def split_by_site(npz_files: list[Path], val_fraction: float, seed: int) -> dict[Path, str]:
    """Assign each site (not each image) entirely to 'train' or 'val', so nearly-identical repeat visits to the same site never leak across the split."""
    sites = sorted({site_from_filename(p) for p in npz_files})
    shuffled = list(sites)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    val_sites = set(shuffled[:n_val])
    return {p: ("val" if site_from_filename(p) in val_sites else "train") for p in npz_files}


def build_coco_dataset(input_path: Path, output_dir: Path, val_fraction: float = 0.2, seed: int = 42) -> dict:
    """
    Convert a Coast Train Landsat-8 zip/folder into a Roboflow-importable COCO
    dataset with a binary sediment/not_sediment class, split by site. Mirrors
    dataset_builder.build_tile_dataset's coco path, reusing the same
    CocoDatasetWriter so the output is consistent with our own coco_export.py.
    """
    npz_files = find_npz_files(input_path)
    split_by_file = split_by_site(npz_files, val_fraction, seed)

    class_id_map = {0: 0, 1: 1}  # binary label values already match final COCO category ids
    writers = {"train": coco_export.CocoDatasetWriter(CLASS_NAMES), "val": coco_export.CocoDatasetWriter(CLASS_NAMES)}
    counts = {"train_images": 0, "val_images": 0, "sites": len({site_from_filename(p) for p in npz_files})}

    for npz_path in tqdm(npz_files, desc="Converting Coast Train samples", unit="image"):
        split = split_by_file[npz_path]
        image, binary_label = load_binary_sediment_label(npz_path)
        height, width = binary_label.shape

        file_name = f"{npz_path.stem}.png"
        coco_export.write_tile_image(output_dir, split, file_name, np.transpose(image, (2, 0, 1)))
        writers[split].add_tile(file_name, width, height, binary_label, class_id_map)

        counts[f"{split}_images"] += 1

    for split, writer in writers.items():
        writer.write(output_dir / split / "_annotations.coco.json")

    return counts
