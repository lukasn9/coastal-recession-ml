# Workflows

A practical guide to every command in this project: what it does, when to run it, and what each parameter means. All commands run through `main.py` from the repository root, after `pip install -r requirements.txt`. Every command also accepts `--help` directly for the same information.

Square brackets mean a parameter is optional; everything else is required. `<placeholders>` mark values you fill in yourself.

## Pipeline order

The stages run in this order for any new region: download, then preprocess, then (optionally) refine-sand, then build-dataset, then train, then predict. `prepare-coco` and `prepare-coasttrain` are side branches that feed back into `build-dataset`'s output or directly into training, used when data is coming from Roboflow instead of being built locally.

```
download → preprocess → refine-sand → build-dataset → train → predict
                              ↑
              prepare-coco ───┘───── prepare-coasttrain
```

`refine-sand` needs a trained sediment model to run, which itself comes from training on a `prepare-coasttrain` (or Roboflow) dataset, so the first time through, you'd train that model before it's available to plug back into this step.

---

## 1. Download imagery for a region

```
python main.py download --region <region> [--satellite <satellite>] [--start-date <date>] [--end-date <date>] [--max-cloud <percent>] [--frequency <period>] [--max-scenes <n>] [--no-thin] [--dry-run]
```

Searches the Microsoft Planetary Computer STAC catalog for scenes over a region's bounding box, thins them down to one scene per time period, and downloads each band clipped to that bounding box (not the full scene) into `datasets/<region>/<satellite>/`. Safe to re-run: it skips any scene that's already fully downloaded.

- `<region>`: one of the keys defined in `configs/regions.yaml` (currently `mediterranean`, `california`, `west_africa`, `southeast_asia`).
- `<satellite>` (`--satellite`): one of the keys defined in `configs/satellites.yaml`. `landsat` (default) is 30m/pixel with an archive back to 2013. `sentinel2` is 10m/pixel from 2016 onward, roughly 9x the pixel count per scene. Every later stage takes the same `--satellite` flag and reads/writes under `datasets/<region>/<satellite>/`, so a region can have both downloaded side by side without collisions.
- `<date>` (`--start-date`, `--end-date`): search window, `YYYY-MM-DD`. `--end-date` defaults to `2024-12-31`; `--start-date` defaults to the chosen satellite's archive start date.
- `<percent>` (`--max-cloud`): maximum scene-wide cloud cover percentage to accept. Default `10`.
- `<period>` (`--frequency`): how densely to sample over time: `monthly`, `seasonal`, `biyearly`, or `yearly`. Within each period, the scene with the lowest cloud cover is kept. Default `monthly`.
- `<n>` (`--max-scenes`): cap the download at this many scenes after thinning. Omit for no cap.
- `--no-thin`: skip the thinning step and download every scene that matches the date range and cloud filter.
- `--dry-run`: print what would be downloaded without downloading anything.

## 2. Preprocess: cloud mask, spectral indices, auto-label

```
python main.py preprocess --region <region> [--satellite <satellite>] [--water-threshold <ndwi>] [--veg-threshold <ndvi>] [--built-threshold <ndbi>] [--overwrite]
```

For every downloaded scene in a region, builds a cloud mask, computes NDWI/NDVI/NDBI, auto-labels each pixel as water, vegetation, or bare from NDWI/NDVI, and writes an RGB composite. Output goes into `datasets/<region>/<satellite>/<scene>/processed/`. Safe to re-run: already-processed scenes are skipped unless `--overwrite` is passed.

Cloud masking and reflectance scaling are satellite-specific under the hood: Landsat decodes the `qa_pixel` band's QA_PIXEL bit flags and uses USGS's fixed scale/offset; Sentinel-2 reads the same file as its SCL classification band and uses ESA's DN/10000 scaling, correcting for the +1000 DN radiometric offset baseline-04.00 scenes (captured from 25 Jan 2022 on) carry, using the `processing_baseline` recorded in each scene's `meta.json` at download time. Sentinel-2's swir16 and qa_pixel bands are natively 20m against the other bands' 10m; they're resampled onto the 10m grid automatically (nearest-neighbor for qa_pixel, since it's a class ID, bilinear for swir16).

- `<region>`: same region keys as above.
- `<satellite>` (`--satellite`): which downloaded satellite's scenes to preprocess (default `landsat`).
- `<ndwi>` (`--water-threshold`): NDWI value above which a pixel counts as water. Default `0.0`.
- `<ndvi>` (`--veg-threshold`): NDVI value above which a pixel counts as vegetation. Default `0.2`.
- `<ndbi>` (`--built-threshold`): NDBI value above which a non-water, non-vegetation pixel counts as built rather than bare. Omitted by default, meaning the built class is off: on real data, NDBI fired just as strongly on bright dry sand as on actual buildings, so enabling it would mislabel sand rather than clean up the bare class. Pass a value (e.g. `0.0`) to try it anyway.
- `--overwrite`: reprocess scenes that already have output, instead of skipping them.

## 3. Refine bare into sand/bare using a trained model

```
python main.py refine-sand --region <region> [--satellite <satellite>] --model <path> [--overwrite]
```

NDWI/NDVI can tell water and vegetation apart cleanly, but everything else just falls into a catch-all `bare` class covering sand along with rock, salt flats, and anything else non-vegetated. This step narrows that down: it runs a trained binary sediment/not_sediment model (see recipe below for how to get one) over the whole scene, but only ever uses its prediction on pixels already labeled `bare`, every other pixel (water, vegetation, built, invalid) is left completely untouched, regardless of what the model says there. Bare pixels the model calls sediment become a new `sand` class; the rest stay `bare`. Writes `label_refined.tif` alongside the original `label.tif`, so the plain NDWI/NDVI labeling stays available for comparison. `build-dataset` automatically prefers `label_refined.tif` over `label.tif` when both exist.

- `<region>`: same region keys as above.
- `<satellite>` (`--satellite`): which preprocessed satellite's scenes to refine (default `landsat`). The sediment model itself was trained on Landsat-derived RGB composites; running it on a different satellite's imagery hasn't been validated.
- `<path>` (`--model`): path to a trained sediment/not_sediment `.pt` checkpoint.
- `--overwrite`: re-refine scenes that already have a `label_refined.tif`, instead of skipping them.

## 4. Build a training dataset

```
python main.py build-dataset --region <region> [--satellite <satellite>] [--export-format <format>] [--tile-size <pixels>] [--val-fraction <fraction>] [--min-valid-fraction <fraction>] [--seed <seed>]
```

Cuts every preprocessed scene in a region into fixed-size tiles and exports them for training. Scenes are split whole into train/val (never split within a scene), so neighboring tiles from the same image can't leak across the split. Each run creates a new numbered `dataset_N` subfolder under `datasets/yolo_datasets/<region>/<satellite>/<format>/`, so repeated runs never overwrite each other and different satellites' tiles (which cover very different real-world distances at the same pixel size) never mix; the exact path is printed when it runs.

- `<region>`: same region keys as above.
- `<satellite>` (`--satellite`): which preprocessed satellite's scenes to tile (default `landsat`).
- `<format>` (`--export-format`): `ultralytics` (images + class-ID mask PNGs, plus `data.yaml`, ready for local `main.py train`) or `coco` (images + RLE-encoded `_annotations.coco.json`, ready to upload to Roboflow). Default `ultralytics`.
- `<pixels>` (`--tile-size`): tile size in pixels, square. Default `640`. If a region's scenes come out smaller than this in either dimension (a tightly cropped bounding box can do that), no tiles will be produced, use a smaller value.
- `<fraction>` (`--val-fraction`): fraction of scenes (not tiles) held out for validation. Default `0.2`.
- `<fraction>` (`--min-valid-fraction`): minimum fraction of a tile that must be clear of cloud/fill to keep it. Default `0.5`.
- `<seed>` (`--seed`): random seed for the scene-level train/val split. Default `42`.

## 5. Bring a Roboflow-reviewed dataset back for training

```
python main.py prepare-coco --input-dir <path> --output-dir <path>
```

Converts a COCO-format segmentation dataset, either our own `build-dataset --export-format coco` output or a dataset exported back out of Roboflow after review, into the `ultralytics` layout that `main.py train` expects. Reads the class list straight out of the COCO file itself rather than assuming it, since Roboflow can renumber categories on export.

- `<path>` (`--input-dir`): root of the COCO dataset. Must contain a `train/` folder and a `val/` or `valid/` folder, each holding its own `_annotations.coco.json`.
- `<path>` (`--output-dir`): project directory. Each run creates a new numbered `dataset_N` subfolder inside it, with `images/`, `masks/`, and `data.yaml`, so repeated runs never overwrite each other.

## 6. Prepare the Coast Train sediment dataset

```
python main.py prepare-coasttrain --input <path> --output-dir <path> [--val-fraction <fraction>] [--seed <seed>]
```

Converts a Coast Train Landsat-8 export (USGS's coastal land-cover dataset) into a COCO dataset with a binary `sediment` / `not_sediment` class, ready to upload to Roboflow. Splits by coastal site rather than by image, so repeat visits to the same site can't leak across train/val. This is a separate, narrower dataset used only to train a sand classifier, not a replacement for the region pipeline above.

- `<path>` (`--input`): path to the Coast Train zip (e.g. `Landsat8_11_001.zip`) or an already-extracted folder. Both work, the zip is unpacked automatically.
- `<path>` (`--output-dir`): project directory. Each run creates a new numbered `dataset_N` subfolder inside it, so repeated runs never overwrite each other.
- `<fraction>` (`--val-fraction`): fraction of sites (not images) held out for validation. Default `0.2`.
- `<seed>` (`--seed`): random seed for the site-level train/val split. Default `42`.

## 7. Train a model

```
python main.py train --data <path> [--model <checkpoint>] [--epochs <n>] [--imgsz <pixels>] [--batch <n>] [--device <device>] [--seed <seed>] [--project <path>] [--name <name>]
```

Trains a YOLO26 semantic segmentation model on a `data.yaml` produced by either `build-dataset` or `prepare-coco`/`prepare-coasttrain`. Identical code path locally and on Colab, only the device differs.

- `<path>` (`--data`): path to a `data.yaml`.
- `<checkpoint>` (`--model`): checkpoint or model config to start from. Default `yolo26n-sem.pt`. Use a larger config (e.g. `yolo26m-sem.pt`) for more capacity at the cost of speed.
- `<n>` (`--epochs`): training epochs. Default `100`.
- `<pixels>` (`--imgsz`): training image size. Default `640`, should generally match the tile size the dataset was built with.
- `<n>` (`--batch`): batch size. Default `16`.
- `<device>` (`--device`): torch device, e.g. `0` for the first CUDA GPU, `mps` for Apple Silicon, `cpu`. Auto-detected (CUDA, then MPS, then CPU) if omitted.
- `<seed>` (`--seed`): random seed. Default `42`.
- `<path>` (`--project`): output directory for run folders. Default `runs/train`.
- `<name>` (`--name`): name for this run's subfolder under `--project`. Default `coastal-seg`.

## 8. Run inference on new images

```
python main.py predict --model <path> --input <path> --output-dir <path>
```

Runs a trained checkpoint on one image, a folder of images, or a zip file, and writes a predicted class-ID mask, a colorized visualization, and a `results.csv` with per-class percentages for every image. Each run creates a new numbered subfolder so repeated runs never overwrite each other.

- `<path>` (`--model`): path to a trained `.pt` checkpoint.
- `<path>` (`--input`): a single image file, a folder, or a `.zip` (unpacked automatically). If the resolved folder contains a `test` subfolder, images are read from there; otherwise every image directly in the folder is used.
- `<path>` (`--output-dir`): project folder. Each run writes into a new `inference_N` subfolder inside it, containing `masks/`, `overlays/`, and `results/results.csv`.

## 9. Set Roboflow credentials

```
python main.py roboflow-config [--api-key <key>] [--workspace <name>]
```

Saves credentials used by `upload-roboflow` to `.roboflow/config.json`, created automatically on first use. That file is gitignored and never committed. Only needed once per machine, or whenever credentials change.

- `<key>` (`--api-key`): your Roboflow API key, from `app.roboflow.com/account/api`.
- `<name>` (`--workspace`): workspace name. Optional, defaults to the workspace tied to your API key; only needed if your account has more than one.

## 10. Upload a dataset to Roboflow

```
python main.py upload-roboflow --dataset-dir <path> --project <name> [--project-type <type>] [--batch-name <name>]
```

Uploads a COCO-format dataset (from `build-dataset --export-format coco` or `prepare-coasttrain`) to a Roboflow project, creating it if it doesn't exist. Requires `roboflow-config` to have been run first. Our own `val/` split folder needs to be seen as `valid/` for Roboflow to recognize it correctly; that rename happens automatically on a temporary copy and never touches your local files.

The upload itself completes before this command waits and polls for Roboflow to finish processing it, so a failure during that wait (including a rate limit error from Roboflow's status endpoint) doesn't mean the upload failed. Transient failures there are retried automatically; if it still gives up, the task_id printed when the upload was sent can be checked again later without re-uploading anything:

```
python main.py upload-roboflow --check-task-id <task_id>
```

- `<path>` (`--dataset-dir`): the `dataset_N` folder itself, containing `train/` and `val(id)/` subfolders.
- `<name>` (`--project`): Roboflow project name.
- `<type>` (`--project-type`): only used if the project doesn't exist yet. Default `instance-segmentation`, the project type that correctly imports our COCO segmentation masks.
- `<name>` (`--batch-name`): name for this upload's batch, shown in Roboflow. Default is auto-generated.
- `<task_id>` (`--check-task-id`): skip uploading and just check a previous upload's status by its task_id.

---

## End-to-end recipes

### Full walkthrough: raw imagery to inference results

This runs every stage once, start to finish, for a single region. Replace `mediterranean` with whichever region key you're working on.

**1. Download imagery.**

```
python main.py download --region mediterranean
```

Searches Landsat 8/9 scenes over the region's bounding box and downloads them, clipped to that box, into `datasets/mediterranean/landsat/`. This can take a while the first time, it's pulling real satellite imagery. Pass `--satellite sentinel2` for 10m imagery instead; every later stage below takes the same flag.

**2. Preprocess it.**

```
python main.py preprocess --region mediterranean
```

For every downloaded scene, builds a cloud mask, computes NDWI/NDVI, auto-labels each pixel water/vegetation/bare, and writes an RGB composite. Output goes into `datasets/mediterranean/landsat/<scene>/processed/`.

**3. (Optional) refine sand.**

```
python main.py refine-sand --region mediterranean --model <path to a trained sediment model>
```

If you already have a trained sediment/not_sediment model (see the Coast Train recipe below for how to get one), this splits the bare class further into sand/bare. Skip this step if you don't have one yet, everything downstream works fine without it, just with a 4-class water/vegetation/bare/built scheme instead of 5.

**4. Build the training dataset.**

```
python main.py build-dataset --region mediterranean
```

Cuts every preprocessed scene into 640x640 tiles and writes them out for training. Prints the exact path it wrote to, something like `datasets/yolo_datasets/mediterranean/landsat/ultralytics/dataset_1/`, use that exact path in the next step rather than assuming `dataset_1`, since a second run would create `dataset_2` instead.

**5. Train.**

```
python main.py train --data datasets/yolo_datasets/mediterranean/landsat/ultralytics/dataset_1/data.yaml
```

Trains a YOLO26 semantic segmentation model. Locally this uses MPS (Apple Silicon) or CPU; on Colab (`notebooks/train_colab.ipynb`) it's the same code, running on a GPU instead. When it finishes, the trained weights are at `runs/train/coastal-seg/weights/best.pt`.

**6. Run inference on new images.**

```
python main.py predict --model runs/train/coastal-seg/weights/best.pt --input <path to an image or folder of images> --output-dir predictions
```

Writes a predicted class mask, a colored visualization, and a `results.csv` of per-class percentages for each image into a new `predictions/inference_1/` folder.

### Review a dataset in Roboflow before training

```
python main.py build-dataset --region <region> --export-format coco
```

Upload the resulting `datasets/yolo_datasets/<region>/<satellite>/coco/dataset_N/` folder to Roboflow (either via `upload-roboflow`, below, or by dragging it into the Roboflow web UI), review or correct it there, then export it back out as COCO and convert it for training:

```
python main.py upload-roboflow --dataset-dir datasets/yolo_datasets/<region>/<satellite>/coco/dataset_N --project <project name>
```

```
python main.py prepare-coco --input-dir <path to Roboflow export> --output-dir <path>
python main.py train --data <path>/data.yaml
```

### Train a sediment (sand) classifier from Coast Train

```
python main.py prepare-coasttrain --input <path to Coast Train zip> --output-dir <path>
```

Upload the resulting COCO dataset to Roboflow (`upload-roboflow` or the web UI), export it back out, then convert and train the same way as any other Roboflow dataset:

```
python main.py prepare-coco --input-dir <path to Roboflow export> --output-dir <path>
python main.py train --data <path>/data.yaml --model yolo26m-sem.pt
```

Before trusting a newly trained sediment model in the pipeline, check its accuracy against real ground truth it never saw during training or model selection, not the dataset's own `val` split, since that was already used to pick the best checkpoint. A Roboflow export's `test/` split (if you uploaded one) works well for this, since `prepare-coco` never touches it.

### Refine sand and re-export to Roboflow

```
python main.py refine-sand --region <region> --model <path to trained sediment model>
python main.py build-dataset --region <region> --export-format coco
```

`build-dataset` automatically prefers `label_refined.tif` over `label.tif`, so this produces a 5-class dataset (water/vegetation/bare/built/sand) ready to upload to Roboflow, without needing any extra flags.

---

## Colab notebooks

`notebooks/train_colab.ipynb` and `notebooks/predict_colab.ipynb` mirror `train` and `predict` respectively, calling the same underlying code, for running on Colab's free GPU instead of locally. Each notebook cell is documented inline.
