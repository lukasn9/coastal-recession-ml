import numpy as np

# Landsat Collection 2 QA_PIXEL bit positions (USGS-defined)
QA_BIT_FILL = 0
QA_BIT_DILATED_CLOUD = 1
QA_BIT_CIRRUS = 2
QA_BIT_CLOUD = 3
QA_BIT_CLOUD_SHADOW = 4

_BAD_PIXEL_BITS = (
    (1 << QA_BIT_FILL)
    | (1 << QA_BIT_DILATED_CLOUD)
    | (1 << QA_BIT_CIRRUS)
    | (1 << QA_BIT_CLOUD)
    | (1 << QA_BIT_CLOUD_SHADOW)
)

# Sentinel-2 L2A Scene Classification Layer (SCL) values (ESA Sen2Cor). Unlike
# QA_PIXEL, this isn't a bit flag band, it's a per-pixel class ID.
SCL_NO_DATA = 0
SCL_SATURATED_OR_DEFECTIVE = 1
SCL_CLOUD_SHADOW = 3
SCL_CLOUD_MEDIUM_PROBABILITY = 8
SCL_CLOUD_HIGH_PROBABILITY = 9
SCL_THIN_CIRRUS = 10

_BAD_SCL_VALUES = (
    SCL_NO_DATA,
    SCL_SATURATED_OR_DEFECTIVE,
    SCL_CLOUD_SHADOW,
    SCL_CLOUD_MEDIUM_PROBABILITY,
    SCL_CLOUD_HIGH_PROBABILITY,
    SCL_THIN_CIRRUS,
)


def build_valid_mask(qa_pixel: np.ndarray, satellite: str = "landsat") -> np.ndarray:
    """
    True where a pixel is clear of fill/cloud/cirrus/cloud-shadow.

    For satellite="landsat", qa_pixel is the QA_PIXEL bit-flag band. For
    satellite="sentinel2", it's the SCL classification band instead (stored
    under the same qa_pixel.tif name, see configs/satellites.yaml).
    """
    if satellite == "landsat":
        return (qa_pixel.astype("uint32") & _BAD_PIXEL_BITS) == 0
    if satellite == "sentinel2":
        return ~np.isin(qa_pixel, _BAD_SCL_VALUES)
    raise ValueError(f"Unknown satellite: {satellite!r}")
