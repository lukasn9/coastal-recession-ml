import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pystac_client
import planetary_computer
import rasterio
from rasterio.env import Env
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.regions import load_regions
from src.satellites import DEFAULT_SATELLITE, load_satellites
from src.scene_meta import write_scene_meta

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Groups scenes into a period key; thinning keeps the lowest-cloud scene per group.
FREQUENCY_KEYS = {
    "monthly": lambda d: f"{d.year}-{d.month:02d}",
    "seasonal": lambda d: f"{d.year}-Q{(d.month - 1) // 3 + 1}",
    "biyearly": lambda d: f"{d.year}-H{1 if d.month <= 6 else 2}",
    "yearly": lambda d: f"{d.year}",
}

# Avoids extra directory-listing/HEAD requests when reading remote COGs.
GDAL_HTTP_ENV = Env(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
)


def thin_to_best(items: list, frequency: str) -> list:
    """Keep one scene per period (monthly/seasonal/biyearly/yearly), lowest cloud cover wins."""
    key_fn = FREQUENCY_KEYS[frequency]
    records = [
        {
            "item": item,
            "date": item.datetime,
            "cloud_cover": item.properties.get("eo:cloud_cover", 100),
        }
        for item in items
    ]
    df = pd.DataFrame(records)
    df["period"] = df["date"].apply(key_fn)
    best = (
        df.sort_values("cloud_cover")
        .groupby("period")
        .first()
        .reset_index()
    )
    return list(best["item"])


def download_clipped_band(href: str, dest: Path, bbox_wgs84: list, max_retries: int = 4) -> None:
    """
    Download only the pixel window covering bbox_wgs84, via a windowed COG read.

    Retries the open+read with exponential backoff on RasterioIOError: reading a
    remote COG over HTTP occasionally fails mid-strip with an error like
    "Cannot read offset/size for strile", a transient blob-storage read glitch
    rather than real file corruption, and a single flaky band shouldn't kill an
    otherwise multi-hour download run. If every attempt fails, the band is
    skipped (not written) rather than raising, same as the no-overlap and
    empty-window cases below, so the scene just registers as incomplete
    downstream instead of crashing the run.
    """
    if dest.exists():
        tqdm.write(f"    skip (exists): {dest.name}")
        return

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(href) as src:
                left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_wgs84)
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                window = window.round_offsets().round_lengths()
                window = window.intersection(Window(0, 0, src.width, src.height))

                if window.width <= 0 or window.height <= 0:
                    tqdm.write(f"    skip (no overlap with bbox): {dest.name}")
                    return

                data = src.read(1, window=window)
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=window.height,
                    width=window.width,
                    transform=src.window_transform(window),
                    compress="deflate",
                )
            break
        except rasterio.errors.RasterioIOError as e:
            if attempt == max_retries:
                tqdm.write(f"    skip (read failed after {max_retries} attempts): {dest.name}, {e}")
                return
            backoff = 2**attempt
            tqdm.write(f"    read failed ({e}), retrying in {backoff}s ({attempt}/{max_retries})...")
            time.sleep(backoff)

    if not data.any():
        # 0 is the fill value for both Landsat SR bands and Sentinel-2 DN
        # bands, so an all-zero window means the bbox falls in a gap of
        # this particular scene's footprint (e.g. a Sentinel-2 MGRS tile
        # edge), not real data. Leaving the file unwritten makes the scene
        # register as incomplete downstream instead of silently processing
        # an empty band.
        tqdm.write(f"    skip (empty window, no data at this bbox): {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data, 1)
    tqdm.write(f"    saved (clipped): {dest.name}")


def print_summary(items: list, region: dict, frequency: str, bands: list) -> None:
    """Print a human-readable summary of scenes to be downloaded."""
    dates = sorted(item.datetime for item in items)
    cloud_covers = [item.properties.get("eo:cloud_cover", None) for item in items]
    cloud_covers = [c for c in cloud_covers if c is not None]

    print("\n--- Scene summary ---")
    print(f"  Region:        {region['name']}")
    print(f"  Frequency:     {frequency}")
    print(f"  Scenes:        {len(items)}")
    print(f"  Date range:    {dates[0].date()} → {dates[-1].date()}")
    print(f"  Cloud cover:   min {min(cloud_covers):.1f}%  "
          f"mean {sum(cloud_covers)/len(cloud_covers):.1f}%  "
          f"max {max(cloud_covers):.1f}%")
    print(f"  Bands:         {', '.join(bands)}")
    print(f"  Est. files:    {len(items) * len(bands)} GeoTIFFs (clipped to region bbox)")
    print("---------------------\n")


def main():
    regions = load_regions()
    satellites = load_satellites()

    parser = argparse.ArgumentParser(
        description="Download satellite scenes for a coastal region, clipped to its bbox."
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
        help="Satellite/collection defined in configs/satellites.yaml (default: landsat, "
        "30m resolution, smaller downloads; sentinel2 gives 10m resolution from 2016 onward, "
        "at roughly 9x the pixel count per scene)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date for search (default: the chosen satellite's archive start date)",
    )
    parser.add_argument(
        "--end-date",
        default="2024-12-31",
        help="End date for search (default: 2024-12-31)",
    )
    parser.add_argument(
        "--max-cloud",
        type=int,
        default=10,
        metavar="PCT",
        help="Maximum cloud cover percentage (default: 10)",
    )
    parser.add_argument(
        "--frequency",
        choices=list(FREQUENCY_KEYS.keys()),
        default="monthly",
        help="Scene sampling frequency: keep the lowest-cloud scene per period (default: monthly)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        metavar="N",
        help="Cap download at N scenes after thinning (default: all)",
    )
    parser.add_argument(
        "--no-thin",
        action="store_true",
        help="Skip frequency thinning and download all matching scenes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scene summary without downloading anything",
    )
    args = parser.parse_args()

    region = regions[args.region]
    satellite = satellites[args.satellite]
    bands = satellite["bands"]
    start_date = args.start_date or satellite["default_start_date"]
    out_dir = DATASETS_DIR / args.region / args.satellite

    print(f"Region:    {region['name']}")
    print(f"Satellite: {satellite['name']} ({satellite['resolution_m']}m/pixel)")
    print(f"Bbox:      {region['bbox']}")
    print(f"Output:    {out_dir}")
    print(f"Dates:     {start_date} → {args.end_date}")
    print(f"Max cloud: {args.max_cloud}%")
    print(f"Frequency: {args.frequency}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results = catalog.search(
        collections=[satellite["collection"]],
        bbox=region["bbox"],
        datetime=f"{start_date}/{args.end_date}",
        query={
            "eo:cloud_cover": {"lt": args.max_cloud},
            **satellite["query"],
        },
    )

    items = []
    for item in tqdm(results.items(), desc="Querying STAC catalog", unit="scene"):
        items.append(item)
    print(f"Found {len(items)} scenes before thinning")

    if not args.no_thin:
        items = thin_to_best(items, args.frequency)
        print(f"Thinned to {len(items)} scenes (one per {args.frequency} period, lowest cloud cover)")

    if args.max_scenes is not None:
        items = items[: args.max_scenes]
        print(f"Capped at {len(items)} scenes (--max-scenes {args.max_scenes})")

    if not items:
        print("No scenes matched, try relaxing --max-cloud or widening the date range.")
        return

    print_summary(items, region, args.frequency, list(bands.keys()))

    if args.dry_run:
        print("Dry run complete, no files downloaded.")
        return

    skipped = 0
    with GDAL_HTTP_ENV:
        progress = tqdm(items, desc="Downloading scenes", unit="scene")
        for item in progress:
            scene_dir = out_dir / item.id
            existing_files = list(scene_dir.glob("*.TIF")) + list(scene_dir.glob("*.tif"))
            if scene_dir.exists() and len(existing_files) >= len(bands):
                tqdm.write(f"skip (complete): {item.id}")
                skipped += 1
                continue

            progress.set_postfix_str(item.id)
            tqdm.write(f"{item.id} ({item.datetime.date()}, {item.properties.get('eo:cloud_cover', '?')}% cloud)")
            signed = planetary_computer.sign(item)
            for band, asset_key in bands.items():
                if asset_key not in signed.assets:
                    tqdm.write(f"    missing asset: {asset_key}")
                    continue
                href = signed.assets[asset_key].href
                download_clipped_band(href, scene_dir / f"{band}.tif", region["bbox"])

            # processing_baseline matters for sentinel2's reflectance conversion (see
            # spectral_indices.to_reflectance); it's None for landsat and simply omitted.
            meta = {"satellite": args.satellite}
            processing_baseline = item.properties.get("s2:processing_baseline")
            if processing_baseline is not None:
                meta["processing_baseline"] = processing_baseline
            write_scene_meta(scene_dir, **meta)

    print(f"\nDone. {len(items) - skipped} downloaded, {skipped} skipped (already complete). Output: {out_dir}")


if __name__ == "__main__":
    main()
