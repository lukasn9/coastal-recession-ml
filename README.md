# Coastal recession from satellite imagery

This repository holds a pipeline for measuring sandy beach erosion (shoreline recession) from Landsat satellite imagery using semi-supervised deep learning. The aim is to detect where coastlines have lost sand between 2013 and 2024 and to estimate a rate of change in meters per year, using freely available multispectral imagery rather than commercial high resolution sources.

## Motivation

Coastal erosion is not a uniform global trend. Luijendijk et al. (2018) found that 24% of the world's sandy beaches erode at more than 0.5 m/year, while 28% accrete and the rest stay roughly stable. Vousdoukas et al. (2020) projected that up to half of the world's sandy beaches could disappear by 2100 without additional adaptation. Nawarat et al. (2024) reported that a third of sandy coastline is already hardened by engineered structures, and that under a high emissions scenario up to 26% of sandy coastline could face severe loss by the end of the century.

Because erosion behaves differently from one coast to another, this project studies a small number of specific regions instead of averaging over the whole planet. Four are currently defined, each chosen because it is documented in the literature as an active erosion or accretion site: the Nile Delta coast in Egypt, Southern California, the Gulf of Guinea coast of Ghana and Togo, and the Mekong Delta in Vietnam. Regions are studied one at a time, and adding a new one means editing a configuration file rather than changing any code.

## Method

The pipeline has three working stages (acquisition, labeling, and model training) and one stage that is planned but not yet built (shoreline extraction and change detection).

### Acquisition

Imagery comes from Landsat 8 and 9 Collection 2 Level 2 surface reflectance products, accessed through the Microsoft Planetary Computer STAC catalog. For a given region, scenes are searched by bounding box, date range, and cloud cover, then thinned to one scene per period, keeping whichever has the lowest cloud cover in that window. The period is configurable (monthly, seasonal, biyearly, or yearly), trading a denser time series against a smaller and faster download.

A full Landsat scene covers roughly 185 by 180 kilometers, far more than most of the regions studied here need. Rather than download whole scenes, each band is requested as a windowed read against the underlying cloud optimized GeoTIFF and clipped to the region's bounding box before being written to disk, so storage scales with the area of interest rather than the scene footprint.

### Labeling

Each scene is cloud masked using the Landsat QA_PIXEL band, converted from raw digital numbers to physical surface reflectance, and used to compute NDWI for water and NDVI for vegetation. Pixels are labeled water or vegetation directly from these indices by thresholding. There is no learned model in this step, which avoids the domain gap that would come from transferring a model trained on a different dataset onto this imagery. Everything that is neither water nor vegetation is labeled bare, a placeholder class covering candidate sand along with anything else non-vegetated, such as bare rock, salt flats, or built structures.

A red, green, blue composite is built separately from the same scene and is the only input the segmentation model ever sees. The spectral indices are used to generate labels, not as model features, so a trained model is not tied to having infrared or shortwave infrared bands available and can in principle run on ordinary color imagery from other sources.

The bare class is the main open problem in the pipeline. NDWI and NDVI separate water and vegetation cleanly, but nothing yet distinguishes sand from other bare land. The planned fix is a refinement step using a model trained on a general land cover dataset, most likely OpenEarthMap, combined with the Segment Anything Model for boundary cleanup, applied only to the bare class rather than to the whole scene.

### Model training

Labeled scenes are cut into fixed size tiles, split by scene rather than by tile, so neighboring tiles from the same image never end up split across training and validation. Tiles are exported in one of two formats: a native layout for Ultralytics YOLO26 semantic segmentation training, with paired PNG images and single channel class ID masks, or a COCO format with run length encoded segmentation, which can be imported into Roboflow for visual review and correction. Both come from the same label rasters, so they stay consistent with each other, and a reviewed COCO export can be converted back into the training layout afterward.

Semantic segmentation was chosen over instance segmentation because water, vegetation, and bare ground are not discrete countable objects. There is no meaningful difference between one patch of water and another, so an architecture built to detect and separate individual instances adds complexity without benefit here. YOLO26 was picked specifically because its segmentation head now supports a genuine semantic segmentation task, distinct from the instance segmentation that earlier YOLO versions offered, which meant staying within a fast, actively maintained framework instead of moving to something like U-Net or DeepLab.

Training runs locally on Apple Silicon through PyTorch's MPS backend, or on a free Google Colab GPU instance. Both call the same training function, so a run started on one machine should behave the same way on the other, aside from ordinary numerical differences between backends.

### Shoreline extraction and change detection

This stage has not been built yet. The plan is to trace the water and land boundary from each scene's segmentation output, sample it along cross shore transects, and fit a linear regression per transect across the full time series to get a recession rate in meters per year. This will be compared against a simpler baseline that thresholds NDWI directly with no learned model involved, so the segmentation model's contribution can be measured rather than assumed.

## Data

Regions are defined in `configs/regions.yaml` by name and bounding box.

| Region key | Description |
|---|---|
| `mediterranean` | Nile Delta coast, Egypt |
| `california` | Southern California coast |
| `west_africa` | Gulf of Guinea coast, Ghana and Togo |
| `southeast_asia` | Mekong Delta, Vietnam |

Downloaded imagery and everything derived from it are not committed to this repository. They are large, easy to regenerate from the scripts here, and specific to whatever regions and time ranges a given user cares about.

## Repository layout

- `configs/regions.yaml`: region definitions, name and bounding box only
- `src/`: importable modules, one concern per file, covering raster I/O, spectral indices, cloud masking, tiling, dataset export, and training
- `scripts/`: thin command line wrappers around `src/`
- `main.py`: single entry point that dispatches to the scripts above by name
- `notebooks/train_colab.ipynb`: Colab notebook for training on a free GPU, calling the same training code as the local script

## Usage

Everything runs through `main.py` from the repository root. Install dependencies first with `pip install -r requirements.txt`.

```
python main.py download --region mediterranean
python main.py preprocess --region mediterranean
python main.py build-dataset --region mediterranean
python main.py train --data datasets/yolo_datasets/mediterranean/ultralytics/data.yaml
```

Each command takes `--help` for its full argument list. A few worth knowing about: `download` accepts `--frequency` to control how densely scenes are sampled over time and `--max-cloud` to set a cloud cover ceiling. `build-dataset` accepts `--export-format`, either `ultralytics` for local training or `coco` for a Roboflow-ready export. A dataset reviewed and re-exported from Roboflow can be converted back with:

```
python main.py prepare-coco --input-dir <path to exported dataset> --output-dir <path>
```

## References

Luijendijk, A., Hagenaars, G., Ranasinghe, R., Baart, F., Donchyts, G., and Aarninkhof, S. (2018). The state of the world's beaches. *Scientific Reports*, 8, 6641.

Vousdoukas, M. I., Ranasinghe, R., Mentaschi, L., Plomaritis, T. A., Athanasiou, P., Luijendijk, A., and Feyen, L. (2020). Sandy coastlines under threat of erosion. *Nature Climate Change*, 10, 260-263.

Nawarat, K., Reyns, J., Vousdoukas, M. I., Duong, T. M., Kras, E., and Ranasinghe, R. (2024). Coastal hardening and what it means for the world's sandy beaches. *Nature Communications*, 15, 10626.

## License

MIT. See `LICENSE`.
