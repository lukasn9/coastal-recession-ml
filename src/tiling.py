import numpy as np


def tile_bounds(height: int, width: int, tile_size: int) -> list[tuple[int, int, int, int]]:
    """
    Non-overlapping (row_start, col_start, row_end, col_end) windows covering an
    array of shape (height, width). Partial edge tiles smaller than tile_size are
    dropped, so every tile is exactly tile_size x tile_size.
    """
    bounds = []
    for row in range(0, height - tile_size + 1, tile_size):
        for col in range(0, width - tile_size + 1, tile_size):
            bounds.append((row, col, row + tile_size, col + tile_size))
    return bounds


def extract_tile(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Slice a tile out of a (H, W) or (bands, H, W) array given (row_start, col_start, row_end, col_end)."""
    row_start, col_start, row_end, col_end = bounds
    if array.ndim == 2:
        return array[row_start:row_end, col_start:col_end]
    return array[:, row_start:row_end, col_start:col_end]
