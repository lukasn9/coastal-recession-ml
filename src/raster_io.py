from pathlib import Path

import numpy as np
import rasterio


def read_band(path: Path) -> tuple[np.ndarray, dict]:
    """Read a single-band GeoTIFF, returning its array and rasterio profile."""
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
    return array, profile


def read_multiband(path: Path) -> tuple[np.ndarray, dict]:
    """Read every band of a multi-band GeoTIFF, returning a (bands, H, W) array and its profile."""
    with rasterio.open(path) as src:
        array = src.read()
        profile = src.profile.copy()
    return array, profile


def read_scene_bands(scene_dir: Path, bands: list[str]) -> tuple[dict[str, np.ndarray], dict]:
    """
    Read multiple bands (e.g. ["red", "green", "blue"]) from a scene folder,
    where each band is stored as "<band>.tif". Returns {band: array} plus the
    shared rasterio profile (CRS/transform), taken from the first band read.
    """
    arrays = {}
    profile = None
    ref_shape = None
    for band in bands:
        array, band_profile = read_band(scene_dir / f"{band}.tif")
        if ref_shape is None:
            ref_shape = array.shape
            profile = band_profile
        elif array.shape != ref_shape:
            raise ValueError(
                f"Band '{band}' has shape {array.shape}, expected {ref_shape} "
                f"(mismatched grid in {scene_dir})"
            )
        arrays[band] = array
    return arrays, profile


def write_geotiff(path: Path, array: np.ndarray, profile: dict, nodata=None) -> None:
    """
    Write a single-band (H, W) or multi-band (bands, H, W) array to disk as a
    GeoTIFF, reusing the CRS/transform from an existing rasterio profile.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_multiband = array.ndim == 3
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=array.shape[0] if is_multiband else 1,
        dtype=array.dtype,
        compress="deflate",
    )
    if nodata is not None:
        out_profile["nodata"] = nodata

    with rasterio.open(path, "w", **out_profile) as dst:
        if is_multiband:
            dst.write(array)
        else:
            dst.write(array, 1)
