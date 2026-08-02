# Coastal recession from satellite imagery

This repository holds a pipeline for measuring sandy beach erosion (shoreline recession) from Landsat satellite imagery using semi-supervised deep learning. The aim is to detect where coastlines have lost sand over the past two decades and to estimate a rate of change in meters per year, using publicly available multispectral imagery.

## Method

The pipeline currently has three working stages (acquisition, labeling, and model training).

### Acquisition

Imagery comes from Landsat 8 and 9 Collection 2 Level 2 surface reflectance products, accessed through the Microsoft Planetary Computer STAC catalog. For a given region, scenes are searched by bounding box, date range, and cloud cover, then thinned to one scene per period, keeping the one that has the lowest cloud cover. The period is configurable (monthly, seasonal, biyearly, or yearly) to adjust dataset size to fit the user's hardware.

### Labeling

Each scene is cloud-masked using the Landsat QA_PIXEL band, converted to physical surface reflectance, and used to compute NDWI for water and NDVI for vegetation. Pixels are labelled water or vegetation directly from these indices by thresholding. Everything that is neither water nor vegetation is labelled bare, a current placeholder class covering candidate sand along with anything else non-vegetated, such as bare rock, salt flats, or built structures.

A red, green, blue composite is built separately from the same scene and is the only input the segmentation model uses. The spectral indices are used to generate labels, so a trained model can function purely with RGB images.

### Model training

Labeled scenes are split into fixed size tiles. Tiles are exported in one of two formats: a native format for Ultralytics YOLO26 semantic segmentation training, or a COCO format with run length encoded segmentation, which can be imported into Roboflow for review and correction.

### Shoreline extraction and change detection

Yet to be implemented.

## Data

Regions are defined in `configs/regions.yaml` by name and bounding box.

| Region key | Description |
|---|---|
| `mediterranean` | Nile Delta coast, Egypt |
| `california` | Southern California coast |
| `west_africa` | Gulf of Guinea coast |
| `southeast_asia` | Mekong Delta, Vietnam |

Downloaded imagery and derived data is not available in this repository and must be obtained and processed.

## Repository layout

- `configs/regions.yaml`: region definitions, name and bounding box only
- `src/`: importable modules, one concern per file, covering raster I/O, spectral indices, cloud masking, tiling, dataset export, and training
- `scripts/`: thin command line wrappers around `src/`
- `main.py`: single entry point that dispatches to the scripts above by name
- `notebooks/train_colab.ipynb`: Colab notebook for training on a free GPU, calling the same training code as the local script
- `notebooks/predict_colab.ipynb`: Colab notebook for testing a trained model on new images, calling the same inference code as the local script

## Usage

After installing dependencies with `pip install -r requirements.txt`, run `main.py` with CLI arguments.

```
python main.py download --region mediterranean
python main.py preprocess --region mediterranean
python main.py build-dataset --region mediterranean
python main.py train --data datasets/yolo_datasets/mediterranean/ultralytics/data.yaml
```

Each command takes `--help` for its full argument list.

```
python main.py prepare-coco --input-dir <path to exported dataset> --output-dir <path>
```

Once a model is trained, `predict` runs it on new images:

```
python main.py predict --model <path to best.pt> --input <path> --output-dir <path>
```

`--input` can be a single image, a folder of images, or a `.zip` file, such as a Roboflow export, which is unpacked automatically. If the resulting folder contains a `test` subfolder, images are read from there. `--output-dir` is a project folder: each run creates a new numbered subfolder inside it (`inference_1`, `inference_2`, ...). Each of those holds a `masks/` folder with the raw class-ID mask per image, an `overlays/` folder with a coloured visualisation, and a `results/results.csv` with one row per image giving the percentage of it taken up by each class.

## References

Luijendijk, A., Hagenaars, G., Ranasinghe, R., Baart, F., Donchyts, G., and Aarninkhof, S. (2018). The state of the world's beaches. *Scientific Reports*, 8, 6641.

Vousdoukas, M. I., Ranasinghe, R., Mentaschi, L., Plomaritis, T. A., Athanasiou, P., Luijendijk, A., and Feyen, L. (2020). Sandy coastlines under threat of erosion. *Nature Climate Change*, 10, 260-263.

Nawarat, K., Reyns, J., Vousdoukas, M. I., Duong, T. M., Kras, E., and Ranasinghe, R. (2024). Coastal hardening and what it means for the world's sandy beaches. *Nature Communications*, 15, 10626.

## License

MIT. See `LICENSE`.
