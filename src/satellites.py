from pathlib import Path

import yaml

SATELLITES_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "satellites.yaml"

DEFAULT_SATELLITE = "landsat"


def load_satellites() -> dict:
    """Load all satellite definitions from configs/satellites.yaml."""
    with open(SATELLITES_CONFIG) as f:
        return yaml.safe_load(f)["satellites"]
