import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
from tqdm import tqdm

from src.preprocessing import REQUIRED_BANDS, process_scene
from src.regions import load_regions
from src.satellites import DEFAULT_SATELLITE, load_satellites
from src.scene_meta import read_scene_meta

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def main():
    regions = load_regions()
    satellites = load_satellites()

    parser = argparse.ArgumentParser(
        description="Preprocess downloaded scenes: cloud mask, NDWI/NDVI, RGB composite, auto-label."
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
        help="Satellite the scenes were downloaded with (default: landsat)",
    )
    parser.add_argument(
        "--water-threshold",
        type=float,
        default=0.0,
        metavar="NDWI",
        help="NDWI threshold above which a pixel is labeled water (default: 0.0)",
    )
    parser.add_argument(
        "--veg-threshold",
        type=float,
        default=0.2,
        metavar="NDVI",
        help="NDVI threshold above which a pixel is labeled vegetation (default: 0.2)",
    )
    parser.add_argument(
        "--built-threshold",
        type=float,
        default=None,
        metavar="NDBI",
        help="NDBI threshold above which a non-water, non-vegetation pixel is labeled "
        "built rather than bare. Off by default: on real data NDBI fired as strongly on "
        "bright dry sand as on actual buildings, so it would mislabel sand rather than "
        "clean up the bare class. Pass a value (e.g. 0.0) to try it anyway.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess scenes even if outputs already exist",
    )
    args = parser.parse_args()

    region_dir = DATASETS_DIR / args.region / args.satellite
    if not region_dir.exists():
        print(f"No downloaded {args.satellite} data for region '{args.region}' at {region_dir}")
        return

    scene_dirs = sorted(
        p for p in region_dir.iterdir()
        if p.is_dir() and all((p / f"{band}.tif").exists() for band in REQUIRED_BANDS)
    )
    print(f"Region: {regions[args.region]['name']}")
    print(f"Satellite: {satellites[args.satellite]['name']}")
    print(f"Found {len(scene_dirs)} downloaded scenes")

    processed, skipped, failed = 0, 0, 0
    for scene_dir in tqdm(scene_dirs, desc="Preprocessing scenes", unit="scene"):
        out_dir = scene_dir / "processed"
        if not args.overwrite and (out_dir / "label.tif").exists():
            tqdm.write(f"skip (already processed): {scene_dir.name}")
            skipped += 1
            continue

        meta = read_scene_meta(scene_dir)
        try:
            process_scene(
                scene_dir,
                out_dir,
                water_threshold=args.water_threshold,
                vegetation_threshold=args.veg_threshold,
                built_threshold=args.built_threshold,
                satellite=args.satellite,
                processing_baseline=meta.get("processing_baseline"),
            )
            tqdm.write(f"processed: {scene_dir.name}")
            processed += 1
        except (FileNotFoundError, rasterio.errors.RasterioIOError) as e:
            tqdm.write(f"skip (incomplete scene): {scene_dir.name}, {e}")
            failed += 1

    print(f"\nDone. {processed} processed, {skipped} skipped, {failed} incomplete.")


if __name__ == "__main__":
    main()
