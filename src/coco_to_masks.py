import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

IGNORE_VALUE = 255

# Mirrors the auto-labeling priority in autolabel.py (water overrides vegetation
# overrides bare) so a Roboflow round-trip paints overlaps the same way our own
# label.tif does. Only applied when a dataset's category names match this scheme
# exactly; otherwise annotations are painted in ascending category-id order.
_KNOWN_CLASS_PRIORITY = ["bare", "vegetation", "water"]  # low to high priority, later wins


def _paint_order(class_names: list[str]) -> list[int]:
    """Dense class ids in the order they should be painted (low to high priority)."""
    lower_names = [n.lower() for n in class_names]
    if set(lower_names) == set(_KNOWN_CLASS_PRIORITY):
        return [lower_names.index(name) for name in _KNOWN_CLASS_PRIORITY]
    return list(range(len(class_names)))


def convert_coco_split(coco_json_path: Path, images_dir: Path, output_images_dir: Path, output_masks_dir: Path) -> list[str]:
    """
    Convert one COCO-annotated split (images + _annotations.coco.json — from our
    own coco_export.py or a Roboflow COCO export) into the Ultralytics semantic-
    segmentation layout: images are copied through unchanged, and each image's
    annotations are painted into a single-channel class-ID PNG mask (pixels with
    no annotation are IGNORE_VALUE). Class ids are read from the COCO file itself
    and remapped to a dense 0..N-1 range — never assumed to already be 0/1/2,
    since Roboflow can renumber categories on export. Returns the class names in
    their assigned dense-id order.
    """
    coco = COCO(str(coco_json_path))
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    class_names = [c["name"] for c in categories]
    coco_id_to_dense_id = {c["id"]: i for i, c in enumerate(categories)}
    paint_order = _paint_order(class_names)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    for image_id, img_info in coco.imgs.items():
        file_name = img_info["file_name"]
        height, width = img_info["height"], img_info["width"]

        anns_by_dense_id: dict[int, list] = {}
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
            dense_id = coco_id_to_dense_id[ann["category_id"]]
            anns_by_dense_id.setdefault(dense_id, []).append(ann)

        mask = np.full((height, width), IGNORE_VALUE, dtype="uint8")
        for dense_id in paint_order:
            for ann in anns_by_dense_id.get(dense_id, []):
                mask[coco.annToMask(ann).astype(bool)] = dense_id

        Image.fromarray(mask, mode="L").save(output_masks_dir / file_name)

        src_image = images_dir / file_name
        dst_image = output_images_dir / file_name
        if src_image.exists() and not dst_image.exists():
            shutil.copy2(src_image, dst_image)

    return class_names
