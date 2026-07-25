import numpy as np

# Landsat Collection 2 Level-2 surface reflectance scaling (USGS-defined):
# reflectance = DN * scale + offset
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


def to_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convert a raw Landsat C2 L2 surface reflectance band (uint16 DN) to physical reflectance."""
    return dn.astype("float32") * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Vegetation is high, bare/water is low or negative."""
    return _normalized_difference(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR). Water is positive, land is negative."""
    return _normalized_difference(green, nir)


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified NDWI = (Green - SWIR) / (Green + SWIR). Sharper water/built-up separation than NDWI."""
    return _normalized_difference(green, swir)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0, 0.0, (a - b) / denom)
    # Landsat L2 reflectance can go slightly negative on dark pixels (deep water,
    # shadow), which pushes the denominator near zero and the ratio outside the
    # mathematically valid [-1, 1] range — clip rather than let it blow up.
    return np.clip(result, -1.0, 1.0).astype("float32")
