import numpy as np

from src import autolabel


def refine_bare_with_sediment_model(label: np.ndarray, rgb_uint8: np.ndarray, model) -> np.ndarray:
    """
    Run a trained binary sediment/not_sediment model over the whole RGB
    composite, but only ever use its prediction to refine pixels that are
    currently labeled bare. Every other pixel (water, vegetation, built,
    invalid) keeps its existing label untouched, regardless of what the model
    predicts there — the model is never trusted outside the region our own
    NDWI/NDVI calculation already flagged as ambiguous. Bare pixels the model
    calls sediment become sand; everything else stays bare.
    """
    sediment_id = next(i for i, name in model.names.items() if name.lower() == "sediment")

    # rgb_uint8 follows this project's (3, H, W) raster convention; ultralytics wants (H, W, 3).
    image = np.transpose(rgb_uint8, (1, 2, 0)) if rgb_uint8.shape[0] == 3 else rgb_uint8

    result = model(image, verbose=False)[0]
    prediction = result.semantic_mask.data
    if hasattr(prediction, "cpu"):
        prediction = prediction.cpu().numpy()
    prediction = np.asarray(prediction)

    refined = label.copy()
    is_bare = label == autolabel.LABEL_BARE
    refined[is_bare & (prediction == sediment_id)] = autolabel.LABEL_SAND
    return refined
