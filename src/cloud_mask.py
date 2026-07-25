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


def build_valid_mask(qa_pixel: np.ndarray) -> np.ndarray:
    """True where a pixel is clear of fill/cloud/cirrus/cloud-shadow, per the QA_PIXEL band."""
    return (qa_pixel.astype("uint32") & _BAD_PIXEL_BITS) == 0
