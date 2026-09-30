import json
from pathlib import Path

META_FILENAME = "meta.json"


def write_scene_meta(scene_dir: Path, **fields) -> None:
    """Write per-scene metadata (satellite, processing baseline, ...) alongside its bands."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    with open(scene_dir / META_FILENAME, "w") as f:
        json.dump(fields, f, indent=2)


def read_scene_meta(scene_dir: Path) -> dict:
    """Read per-scene metadata written at download time. Empty dict if missing."""
    path = scene_dir / META_FILENAME
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
