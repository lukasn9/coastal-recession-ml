import numpy as np


def composite_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Stack three reflectance bands into a (3, H, W) true-color array."""
    return np.stack([red, green, blue], axis=0)


def stretch_to_uint8(reflectance_rgb: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile-stretch a (3, H, W) reflectance array to a viewable uint8 [0, 255] image."""
    out = np.empty_like(reflectance_rgb, dtype="uint8")
    for i in range(reflectance_rgb.shape[0]):
        band = reflectance_rgb[i]
        lo, hi = np.percentile(band, [low_pct, high_pct])
        stretched = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
        out[i] = (stretched * 255).astype("uint8")
    return out
