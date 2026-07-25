from pathlib import Path

import yaml

REGIONS_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "regions.yaml"


def load_regions() -> dict:
    """Load all region definitions from configs/regions.yaml."""
    with open(REGIONS_CONFIG) as f:
        return yaml.safe_load(f)["regions"]
