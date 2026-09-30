import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_builder import EXPORT_FORMATS, build_tile_dataset
from src.regions import load_regions
from src.run_dirs import next_run_dir
from src.satellites import DEFAULT_SATELLITE, load_satellites

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def main():
    regions = load_regions()
    satellites = load_satellites()

    parser = argparse.ArgumentParser(
        description="Build a YOLO semantic-segmentation tile dataset (images + class-ID masks) "
        "from preprocessed scenes."
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
        help="Satellite the scenes were downloaded and preprocessed with (default: landsat)",
    )
    parser.add_argument(
        "--export-format",
        choices=list(EXPORT_FORMATS),
        default="ultralytics",
        help="'ultralytics' for local `yolo semantic train` (default), "
        "'coco' (RLE) for uploading to Roboflow",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Tile size in pixels, square (default: 640)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of scenes (not tiles) held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        metavar="FRAC",
        help="Minimum fraction of a tile that must be clear of cloud/fill to keep it (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the scene-level train/val split (default: 42)",
    )
    args = parser.parse_args()

    region_dir = DATASETS_DIR / args.region / args.satellite
    if not region_dir.exists():
        print(f"No downloaded {args.satellite} data for region '{args.region}' at {region_dir}")
        return

    # Satellites and formats never share a folder (tile_size means a different
    # real-world coverage at 10m vs 30m/pixel), and each run gets its own
    # numbered dataset_N subfolder inside that, so repeated runs never
    # overwrite each other.
    format_dir = DATASETS_DIR / "yolo_datasets" / args.region / args.satellite / args.export_format
    output_dir = next_run_dir(format_dir, prefix="dataset")
    print(f"Region:      {regions[args.region]['name']}")
    print(f"Satellite:   {satellites[args.satellite]['name']}")
    print(f"Format:      {args.export_format}")
    print(f"Tile size:   {args.tile_size}x{args.tile_size}")
    print(f"Val split:   {args.val_fraction:.0%} of scenes (seed={args.seed})")
    print(f"Output:      {output_dir}")

    counts = build_tile_dataset(
        region_dir=region_dir,
        output_dir=output_dir,
        export_format=args.export_format,
        tile_size=args.tile_size,
        val_fraction=args.val_fraction,
        min_valid_fraction=args.min_valid_fraction,
        seed=args.seed,
    )

    print(f"\nDone. {counts['scenes']} scenes processed.")
    print(f"  train tiles: {counts['train_tiles']}")
    print(f"  val tiles:   {counts['val_tiles']}")
    print(f"  skipped (too cloudy/empty): {counts['skipped_tiles']}")


if __name__ == "__main__":
    main()
