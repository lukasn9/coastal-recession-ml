from pathlib import Path

import numpy as np

from src import autolabel, cloud_mask, raster_io, rgb, spectral_indices

REQUIRED_BANDS = ["blue", "green", "red", "nir08", "swir16", "qa_pixel"]


def process_scene(
    scene_dir: Path,
    out_dir: Path,
    water_threshold: float = 0.0,
    vegetation_threshold: float = 0.2,
    built_threshold: float | None = None,
    satellite: str = "landsat",
    processing_baseline: str | None = None,
) -> dict[str, Path]:
    """
    Run the full preprocessing + auto-labeling pipeline on one downloaded scene folder.

    Writes rgb.tif, ndwi.tif, ndvi.tif, ndbi.tif, cloud_mask.tif, label.tif into out_dir
    and returns their paths. This is the single entry point a CLI or a future GUI should
    call per scene, everything it does is composed from the smaller pure functions
    in spectral_indices/cloud_mask/rgb/autolabel, which can also be called directly
    (e.g. to preview one index or re-label with different thresholds).

    ndbi.tif is always written (cheap, and useful on its own), but the built
    class it can produce is off by default (built_threshold=None), see
    autolabel.label_from_indices for why.

    satellite selects reflectance scaling and cloud-mask decoding ("landsat"
    or "sentinel2", see spectral_indices.to_reflectance and
    cloud_mask.build_valid_mask). processing_baseline only matters for
    sentinel2, and should come from that scene's meta.json (src/scene_meta.py).
    For sentinel2, qa_pixel.tif and swir16.tif are natively 20m against the
    other bands' 10m, so read_scene_bands resamples them onto the 10m grid.
    """
    bands, profile = raster_io.read_scene_bands(scene_dir, REQUIRED_BANDS, categorical_bands={"qa_pixel"})

    reflectance = {
        name: spectral_indices.to_reflectance(array, satellite=satellite, processing_baseline=processing_baseline)
        for name, array in bands.items()
        if name != "qa_pixel"
    }

    valid_mask = cloud_mask.build_valid_mask(bands["qa_pixel"], satellite=satellite)
    ndwi = spectral_indices.compute_ndwi(reflectance["green"], reflectance["nir08"])
    ndvi = spectral_indices.compute_ndvi(reflectance["nir08"], reflectance["red"])
    ndbi = spectral_indices.compute_ndbi(reflectance["swir16"], reflectance["nir08"])
    label = autolabel.label_from_indices(
        ndwi,
        ndvi,
        valid_mask,
        ndbi=ndbi,
        water_threshold=water_threshold,
        vegetation_threshold=vegetation_threshold,
        built_threshold=built_threshold,
    )

    rgb_reflectance = rgb.composite_rgb(reflectance["red"], reflectance["green"], reflectance["blue"])
    rgb_uint8 = rgb.stretch_to_uint8(rgb_reflectance)

    ndwi_out = np.where(valid_mask, ndwi, np.nan).astype("float32")
    ndvi_out = np.where(valid_mask, ndvi, np.nan).astype("float32")
    ndbi_out = np.where(valid_mask, ndbi, np.nan).astype("float32")

    outputs = {
        "rgb": out_dir / "rgb.tif",
        "ndwi": out_dir / "ndwi.tif",
        "ndvi": out_dir / "ndvi.tif",
        "ndbi": out_dir / "ndbi.tif",
        "cloud_mask": out_dir / "cloud_mask.tif",
        "label": out_dir / "label.tif",
    }

    raster_io.write_geotiff(outputs["rgb"], rgb_uint8, profile)
    raster_io.write_geotiff(outputs["ndwi"], ndwi_out, profile, nodata=float("nan"))
    raster_io.write_geotiff(outputs["ndvi"], ndvi_out, profile, nodata=float("nan"))
    raster_io.write_geotiff(outputs["ndbi"], ndbi_out, profile, nodata=float("nan"))
    raster_io.write_geotiff(outputs["cloud_mask"], valid_mask.astype("uint8"), profile)
    raster_io.write_geotiff(outputs["label"], label, profile, nodata=autolabel.LABEL_INVALID)

    return outputs
