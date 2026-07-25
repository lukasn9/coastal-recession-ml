from pathlib import Path

import numpy as np

from src import autolabel, cloud_mask, raster_io, rgb, spectral_indices

REQUIRED_BANDS = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]


def process_scene(
    scene_dir: Path,
    out_dir: Path,
    water_threshold: float = 0.0,
    vegetation_threshold: float = 0.2,
) -> dict[str, Path]:
    """
    Run the full preprocessing + auto-labeling pipeline on one downloaded scene folder.

    Writes rgb.tif, ndwi.tif, ndvi.tif, cloud_mask.tif, label.tif into out_dir and
    returns their paths. This is the single entry point a CLI or a future GUI should
    call per scene — everything it does is composed from the smaller pure functions
    in spectral_indices/cloud_mask/rgb/autolabel, which can also be called directly
    (e.g. to preview one index or re-label with different thresholds).
    """
    bands, profile = raster_io.read_scene_bands(scene_dir, REQUIRED_BANDS)

    reflectance = {
        name: spectral_indices.to_reflectance(array)
        for name, array in bands.items()
        if name != "qa_pixel"
    }

    valid_mask = cloud_mask.build_valid_mask(bands["qa_pixel"])
    ndwi = spectral_indices.compute_ndwi(reflectance["green"], reflectance["nir08"])
    ndvi = spectral_indices.compute_ndvi(reflectance["nir08"], reflectance["red"])
    label = autolabel.label_from_indices(ndwi, ndvi, valid_mask, water_threshold, vegetation_threshold)

    rgb_reflectance = rgb.composite_rgb(reflectance["red"], reflectance["green"], reflectance["blue"])
    rgb_uint8 = rgb.stretch_to_uint8(rgb_reflectance)

    ndwi_out = np.where(valid_mask, ndwi, np.nan).astype("float32")
    ndvi_out = np.where(valid_mask, ndvi, np.nan).astype("float32")

    outputs = {
        "rgb": out_dir / "rgb.tif",
        "ndwi": out_dir / "ndwi.tif",
        "ndvi": out_dir / "ndvi.tif",
        "cloud_mask": out_dir / "cloud_mask.tif",
        "label": out_dir / "label.tif",
    }

    raster_io.write_geotiff(outputs["rgb"], rgb_uint8, profile)
    raster_io.write_geotiff(outputs["ndwi"], ndwi_out, profile, nodata=float("nan"))
    raster_io.write_geotiff(outputs["ndvi"], ndvi_out, profile, nodata=float("nan"))
    raster_io.write_geotiff(outputs["cloud_mask"], valid_mask.astype("uint8"), profile)
    raster_io.write_geotiff(outputs["label"], label, profile, nodata=autolabel.LABEL_INVALID)

    return outputs
