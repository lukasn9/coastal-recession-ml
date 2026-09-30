import numpy as np

# Landsat Collection 2 Level-2 surface reflectance scaling (USGS-defined):
# reflectance = DN * scale + offset
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2

# Sentinel-2 L2A digital numbers are reflectance x10000 (QUANTIFICATION_VALUE).
# Processing baseline 04.00, applied to all scenes captured from 25 Jan 2022
# onward, added a uniform +1000 DN shift (BOA_ADD_OFFSET) so that negative
# reflectance over very dark surfaces can be encoded instead of clipped at 0.
# Earlier baselines have no such shift. See ESA's PSD and
# https://forum.step.esa.int/t/info-introduction-of-additional-radiometric-offset-in-pb04-00-products/35431
SENTINEL2_SR_SCALE = 0.0001
SENTINEL2_BOA_ADD_OFFSET = -1000
SENTINEL2_OFFSET_BASELINE = 4.0


def to_reflectance(dn: np.ndarray, satellite: str = "landsat", processing_baseline: str | None = None) -> np.ndarray:
    """
    Convert a raw surface reflectance band (uint16 DN) to physical reflectance.

    processing_baseline only matters for satellite="sentinel2" (see
    SENTINEL2_BOA_ADD_OFFSET above); it is the "processing_baseline" field
    written to a scene's meta.json at download time, e.g. "05.10".
    """
    if satellite == "landsat":
        return dn.astype("float32") * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    if satellite == "sentinel2":
        offset = SENTINEL2_BOA_ADD_OFFSET if _needs_boa_offset(processing_baseline) else 0
        return (dn.astype("float32") + offset) * SENTINEL2_SR_SCALE
    raise ValueError(f"Unknown satellite: {satellite!r}")


def _needs_boa_offset(processing_baseline: str | None) -> bool:
    if processing_baseline is None:
        return False
    return float(processing_baseline) >= SENTINEL2_OFFSET_BASELINE


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Vegetation is high, bare/water is low or negative."""
    return _normalized_difference(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green - NIR) / (Green + NIR). Water is positive, land is negative."""
    return _normalized_difference(green, nir)


def compute_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified NDWI = (Green - SWIR) / (Green + SWIR). Sharper water/built-up separation than NDWI."""
    return _normalized_difference(green, swir)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDBI = (SWIR - NIR) / (SWIR + NIR). Built-up surfaces (concrete, asphalt, roofs) are positive."""
    return _normalized_difference(swir, nir)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denom == 0, 0.0, (a - b) / denom)
    # Landsat L2 reflectance can go slightly negative on dark pixels (deep water,
    # shadow), which pushes the denominator near zero and the ratio outside the
    # mathematically valid [-1, 1] range, clip rather than let it blow up.
    return np.clip(result, -1.0, 1.0).astype("float32")
