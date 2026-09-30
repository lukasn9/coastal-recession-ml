from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


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


def read_scene_bands(
    scene_dir: Path,
    bands: list[str],
    categorical_bands: frozenset[str] = frozenset(),
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Read multiple bands (e.g. ["red", "green", "blue"]) from a scene folder,
    where each band is stored as "<band>.tif". The first band read sets the
    reference grid (CRS/transform/shape); any later band on a different grid
    (e.g. Sentinel-2's 20m swir16/qa_pixel next to its 10m optical bands) is
    resampled onto it rather than rejected, nearest-neighbor for bands listed
    in categorical_bands (e.g. a classification band, where averaging class
    IDs is meaningless), bilinear otherwise. Returns {band: array} plus the
    reference rasterio profile.
    """
    arrays = {}
    profile = None
    for band in bands:
        path = scene_dir / f"{band}.tif"
        with rasterio.open(path) as src:
            if profile is None:
                arrays[band] = src.read(1)
                profile = src.profile.copy()
                continue
            same_grid = (
                src.crs == profile["crs"]
                and src.transform == profile["transform"]
                and src.width == profile["width"]
                and src.height == profile["height"]
            )
            if same_grid:
                arrays[band] = src.read(1)
                continue
            resampling = Resampling.nearest if band in categorical_bands else Resampling.bilinear
            dest = np.empty((profile["height"], profile["width"]), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                resampling=resampling,
            )
            arrays[band] = dest
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
