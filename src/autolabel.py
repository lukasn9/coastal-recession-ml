import numpy as np

LABEL_WATER = 0
LABEL_VEGETATION = 1
LABEL_BARE = 2  # candidate sand + other non-vegetated land — not yet disambiguated
LABEL_INVALID = 255


def label_from_indices(
    ndwi: np.ndarray,
    ndvi: np.ndarray,
    valid_mask: np.ndarray,
    water_threshold: float = 0.0,
    vegetation_threshold: float = 0.2,
) -> np.ndarray:
    """
    Auto-label pixels as water / vegetation / bare from NDWI + NDVI.
    Water takes priority over vegetation at the waterline. 'Bare' covers sand
    and other non-vegetated land alike — it isn't a final sand label, just
    what's left after water and vegetation are removed. Cloud/fill pixels
    (per valid_mask) are marked invalid rather than guessed at.
    """
    labels = np.full(ndwi.shape, LABEL_BARE, dtype="uint8")
    labels[ndvi >= vegetation_threshold] = LABEL_VEGETATION
    labels[ndwi >= water_threshold] = LABEL_WATER
    labels[~valid_mask] = LABEL_INVALID
    return labels
