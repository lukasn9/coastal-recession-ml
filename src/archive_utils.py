import shutil
import tempfile
from pathlib import Path


def extract_if_zip(path: Path, prefix: str = "extracted_") -> Path:
    """If path is a .zip file, unpack it to a fresh temp folder and return that folder. Otherwise return path unchanged."""
    if path.suffix.lower() != ".zip":
        return path
    extract_dir = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.unpack_archive(str(path), str(extract_dir))
    return extract_dir
