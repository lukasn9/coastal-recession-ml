import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio

from src.preprocessing import REQUIRED_BANDS, process_scene
from src.regions import load_regions

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def main():
    regions = load_regions()

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
        "--overwrite",
        action="store_true",
        help="Reprocess scenes even if outputs already exist",
    )
    args = parser.parse_args()

    region_dir = DATASETS_DIR / args.region
    if not region_dir.exists():
        print(f"No downloaded data for region '{args.region}' at {region_dir}")
        return

    scene_dirs = sorted(
        p for p in region_dir.iterdir()
        if p.is_dir() and all((p / f"{band}.tif").exists() for band in REQUIRED_BANDS)
    )
    print(f"Region: {regions[args.region]['name']}")
    print(f"Found {len(scene_dirs)} downloaded scenes")

    processed, skipped, failed = 0, 0, 0
    for i, scene_dir in enumerate(scene_dirs, 1):
        out_dir = scene_dir / "processed"
        if not args.overwrite and (out_dir / "label.tif").exists():
            print(f"[{i}/{len(scene_dirs)}] skip (already processed): {scene_dir.name}")
            skipped += 1
            continue

        try:
            process_scene(scene_dir, out_dir, args.water_threshold, args.veg_threshold)
            print(f"[{i}/{len(scene_dirs)}] processed: {scene_dir.name}")
            processed += 1
        except (FileNotFoundError, rasterio.errors.RasterioIOError) as e:
            print(f"[{i}/{len(scene_dirs)}] skip (incomplete scene): {scene_dir.name} — {e}")
            failed += 1

    print(f"\nDone. {processed} processed, {skipped} skipped, {failed} incomplete.")


if __name__ == "__main__":
    main()
