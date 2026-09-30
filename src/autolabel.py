import numpy as np

LABEL_WATER = 0
LABEL_VEGETATION = 1
LABEL_BARE = 2  # candidate sand + other non-vegetated, non-built, non-sand-confirmed land
LABEL_BUILT = 3  # concrete, asphalt, roofs — carved out of bare via NDBI
LABEL_SAND = 4  # confirmed sand — carved out of bare via a trained sediment classifier
LABEL_INVALID = 255


def label_from_indices(
    ndwi: np.ndarray,
    ndvi: np.ndarray,
    valid_mask: np.ndarray,
    ndbi: np.ndarray | None = None,
    water_threshold: float = 0.0,
    vegetation_threshold: float = 0.2,
    built_threshold: float | None = None,
) -> np.ndarray:
    """
    Auto-label pixels as water / vegetation / bare (/ built) from NDWI, NDVI,
    and optionally NDBI. Water takes priority over vegetation at the
    waterline. Everything left over after water and vegetation are removed
    starts out as 'bare'.

    The built split is opt-in and off by default (built_threshold=None): on a
    real Nile Delta test scene, NDBI turned out to fire just as strongly on
    bright dry sand as on actual buildings (median NDBI ~1.0 over a sand spit
    vs ~-0.28 over the real urban grid nearby), so relying on it by default
    would mislabel sand as built rather than clean up the bare class. Pass a
    real ndbi array and built_threshold to turn it on for testing or for a
    region where it behaves better. 'Bare' still isn't a final sand label
    either way, just what's left after water, vegetation, and (if enabled)
    built surfaces are removed. Cloud/fill pixels (per valid_mask) are marked
    invalid rather than guessed at.
    """
    labels = np.full(ndwi.shape, LABEL_BARE, dtype="uint8")
    labels[ndvi >= vegetation_threshold] = LABEL_VEGETATION
    if ndbi is not None and built_threshold is not None:
        labels[(labels == LABEL_BARE) & (ndbi >= built_threshold)] = LABEL_BUILT
    labels[ndwi >= water_threshold] = LABEL_WATER
    labels[~valid_mask] = LABEL_INVALID
    return labels
