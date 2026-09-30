import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from ultralytics import YOLO

from src import autolabel, raster_io
from src.regions import load_regions
from src.sand_refinement import refine_bare_with_sediment_model
from src.satellites import DEFAULT_SATELLITE, load_satellites

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def main():
    regions = load_regions()
    satellites = load_satellites()

    parser = argparse.ArgumentParser(
        description="Refine the bare class into sand/bare using a trained sediment "
        "binary segmentation model. The model is only ever applied to pixels already "
        "labeled bare by preprocess; every other pixel keeps its existing label untouched. "
        "Writes label_refined.tif alongside label.tif, leaving the original NDWI/NDVI-only "
        "labeling available for comparison."
    )
    parser.add_argument(
        "--region",
        required=True,
        choices=list(regions.keys()),
        help="Region key defined in configs/regions.yaml",
    )
    parser.add_argument(
        "--satellite",
        choices=list(satellites.keys()),
        default=DEFAULT_SATELLITE,
        help="Satellite the scenes were downloaded and preprocessed with (default: landsat). "
        "The sediment model itself was trained on Landsat-derived RGB composites; applying it "
        "to a different satellite's imagery has not been validated.",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a trained sediment/not_sediment .pt checkpoint",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-refine scenes even if label_refined.tif already exists",
    )
    args = parser.parse_args()

    region_dir = DATASETS_DIR / args.region / args.satellite
    if not region_dir.exists():
        print(f"No downloaded {args.satellite} data for region '{args.region}' at {region_dir}")
        return

    scene_dirs = sorted(
        p for p in region_dir.iterdir()
        if p.is_dir() and (p / "processed" / "label.tif").exists()
    )
    print(f"Region: {regions[args.region]['name']}")
    print(f"Satellite: {satellites[args.satellite]['name']}")
    print(f"Found {len(scene_dirs)} preprocessed scenes")

    model = YOLO(str(args.model))

    refined_count, skipped = 0, 0
    for scene_dir in tqdm(scene_dirs, desc="Refining sand", unit="scene"):
        processed_dir = scene_dir / "processed"
        out_path = processed_dir / "label_refined.tif"
        if out_path.exists() and not args.overwrite:
            tqdm.write(f"skip (already refined): {scene_dir.name}")
            skipped += 1
            continue

        label, profile = raster_io.read_band(processed_dir / "label.tif")
        rgb, _ = raster_io.read_multiband(processed_dir / "rgb.tif")

        refined = refine_bare_with_sediment_model(label, rgb, model)
        raster_io.write_geotiff(out_path, refined, profile, nodata=autolabel.LABEL_INVALID)

        tqdm.write(f"refined: {scene_dir.name}")
        refined_count += 1

    print(f"\nDone. {refined_count} refined, {skipped} skipped.")


if __name__ == "__main__":
    main()
