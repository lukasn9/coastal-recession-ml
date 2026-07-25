import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

_KNOWN_COLORS = {
    "water": (46, 111, 191),
    "vegetation": (63, 158, 89),
    "bare": (207, 185, 137),
}
_FALLBACK_PALETTE = [(196, 78, 82), (140, 109, 191), (219, 160, 60), (90, 180, 180), (150, 150, 150)]


def resolve_input(input_path: Path):
    """
    Resolve a user-supplied path into an image (or list of images) ready for
    inference. A single image file is returned as-is. A zip file (e.g. a
    Roboflow export) is unpacked to a temporary folder first. A folder
    containing a 'test' subfolder (the layout inside a Roboflow export) uses
    that subfolder; any other folder uses every image found directly inside it.
    """
    if input_path.suffix.lower() == ".zip":
        extract_dir = Path(tempfile.mkdtemp(prefix="predict_input_"))
        shutil.unpack_archive(str(input_path), str(extract_dir))
        input_path = extract_dir

    if input_path.is_file():
        return input_path

    search_dir = input_path / "test" if (input_path / "test").is_dir() else input_path
    images = sorted(p for p in search_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {search_dir}")
    return images


def next_run_dir(base_dir: Path, prefix: str = "inference") -> Path:
    """
    Pick and create the next numbered run directory under base_dir, e.g.
    inference_1, inference_2, matching how Ultralytics numbers its own
    runs/train folders, so repeated inference runs never overwrite older
    output and the latest one is easy to spot.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{prefix}_"):
            suffix = p.name[len(prefix) + 1:]
            if suffix.isdigit():
                existing.append(int(suffix))

    run_dir = base_dir / f"{prefix}_{max(existing, default=0) + 1}"
    run_dir.mkdir(parents=True)
    return run_dir


def class_colors(names: dict[int, str]) -> dict[int, tuple[int, int, int]]:
    """Pick a display color per class id: known coastal classes get fixed colors, anything else a fallback palette."""
    colors = {}
    fallback_i = 0
    for class_id, name in names.items():
        color = _KNOWN_COLORS.get(name.lower())
        if color is None:
            color = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
            fallback_i += 1
        colors[class_id] = color
    return colors


def colorize_mask(mask: np.ndarray, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Render a predicted class-ID mask as an RGB image for visual inspection."""
    rgb = np.zeros((*mask.shape, 3), dtype="uint8")
    for class_id, color in colors.items():
        rgb[mask == class_id] = color
    return rgb


def class_percentages(mask: np.ndarray, names: dict[int, str]) -> dict[str, float]:
    """Fraction of pixels, as a percentage rounded to 2 decimal places, assigned to each class."""
    total = mask.size
    return {name: round(float((mask == class_id).sum()) / total * 100, 2) for class_id, name in names.items()}


def write_results_csv(summaries: list[dict], path: Path) -> None:
    """Write one row per image (file name plus its per-class percentages) to a CSV."""
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def run_inference(model: YOLO, source, run_dir: Path) -> list[dict]:
    """
    Run the model on a single image path or a list of image paths, writing a
    class-ID mask into run_dir/masks, a colorized visualization into
    run_dir/overlays, and a results.csv into run_dir/results with one row per
    image and its per-class percentages. The mask holds raw class ids
    (0, 1, 2, ...) and looks solid black in a normal viewer; the overlay is
    the one meant for looking at. Returns the same per-image summaries written
    to the CSV, read from the model's own class names so this works for any model.
    """
    masks_dir = run_dir / "masks"
    overlays_dir = run_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    colors = class_colors(model.names)

    # Ultralytics doesn't reliably preserve original file names on result.path
    # for batched (list) sources, so track them ourselves instead, in order.
    source_paths = [source] if isinstance(source, Path) else list(source)

    # stream=True makes predict() a generator that processes and yields one
    # image at a time, instead of running the whole batch and holding every
    # result in memory before we save any of them. Matters for large folders.
    results = model(source_paths, verbose=False, stream=True)

    summaries = []
    for image_path, result in zip(source_paths, results):
        mask = result.semantic_mask.data
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask = np.asarray(mask).astype("uint8")

        stem = image_path.stem
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
        Image.fromarray(colorize_mask(mask, colors)).save(overlays_dir / f"{stem}.png")

        summary = {"file": image_path.name}
        summary.update(class_percentages(mask, model.names))
        summaries.append(summary)

    write_results_csv(summaries, run_dir / "results" / "results.csv")
    return summaries
