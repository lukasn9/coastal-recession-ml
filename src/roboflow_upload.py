import tempfile
import time
from pathlib import Path

from roboflow import Roboflow
from roboflow.adapters import rfapi
from roboflow.adapters.rfapi import RoboflowError

from src.roboflow_config import load_config

# Roboflow's own folder parser infers a split by checking for the substring
# "valid" in each file's path, not "val" (our own convention, used by
# build-dataset and prepare-coasttrain). Left as-is, our val/ folder would be
# silently miscategorized as train, since "valid" never appears in "val".
_SPLIT_RENAMES = {"val": "valid"}


def _prepare_upload_root(dataset_dir: Path, tmp_dir: Path) -> Path:
    """
    Build a symlink tree mirroring dataset_dir, renaming split folders Roboflow
    wouldn't recognize (val -> valid). Symlinks are per-file, not per-directory:
    the zip step walks the tree without following symlinked directories, but
    reads through symlinked files transparently, so this avoids copying
    potentially large datasets just to rename one folder.
    """
    upload_root = tmp_dir / "dataset"
    for split_dir in dataset_dir.iterdir():
        if not split_dir.is_dir():
            continue
        target_dir = upload_root / _SPLIT_RENAMES.get(split_dir.name, split_dir.name)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in split_dir.iterdir():
            if item.is_file():
                (target_dir / item.name).symlink_to(item.resolve())
    return upload_root


def _poll_status_resilient(
    api_key: str,
    workspace_url: str,
    task_id: str,
    poll_interval: float = 20.0,
    poll_timeout: float = 3600.0,
    max_consecutive_errors: int = 8,
    max_backoff: float = 120.0,
) -> dict:
    """
    Poll an async zip upload's status until it completes or fails. Unlike the
    SDK's own polling loop, this tolerates transient errors from the status
    endpoint itself (e.g. 429 Too Many Requests) with exponential backoff,
    instead of crashing on the first one. That matters because a failure here
    only means we couldn't confirm completion, not that the upload failed: the
    actual data transfer already finished before polling starts.

    A large dataset can take Roboflow's backend minutes to process, so the
    default poll_interval is deliberately unhurried rather than checking every
    few seconds, that alone seems to trigger 429s on this endpoint faster than
    the interval would suggest.
    """
    deadline = time.monotonic() + poll_timeout
    consecutive_errors = 0
    last_progress = None
    while True:
        try:
            status = rfapi.get_zip_upload_status(api_key, workspace_url, task_id)
            consecutive_errors = 0
        except RoboflowError as e:
            consecutive_errors += 1
            if consecutive_errors > max_consecutive_errors:
                raise RoboflowError(
                    f"Status check failed {consecutive_errors} times in a row (task_id={task_id}): {e}. "
                    "The upload itself very likely already completed, since this only happens after the "
                    "real data transfer succeeds. Check the Roboflow project directly, or retry the status "
                    "check later with this task_id."
                ) from e
            backoff = min(poll_interval * (2**consecutive_errors), max_backoff)
            print(f"  status check failed ({e}), retrying in {backoff:.0f}s...")
            time.sleep(backoff)
            continue

        state = status.get("status")
        progress = (status.get("progress") or {}).get("current")
        if progress is not None and progress != last_progress:
            print(f"  zip-upload progress: {progress}")
            last_progress = progress
        if state in {"completed", "failed"}:
            return status
        if time.monotonic() >= deadline:
            raise RoboflowError(f"Zip upload polling timed out after {poll_timeout}s (task_id={task_id}, last_status={state}).")
        time.sleep(poll_interval)


def check_upload_status(task_id: str) -> dict:
    """Check an already-submitted upload's status by task_id, without re-uploading anything."""
    config = load_config()
    rf = Roboflow(api_key=config["api_key"])
    workspace = rf.workspace(config.get("workspace") or None)
    return _poll_status_resilient(config["api_key"], workspace.url, task_id)


def upload_dataset(
    dataset_dir: Path,
    project_name: str,
    project_type: str = "instance-segmentation",
    batch_name: str | None = None,
) -> dict | None:
    """
    Upload a COCO-format dataset directory (train/val(id)/test, each with
    images and its own _annotations.coco.json, the layout build-dataset and
    prepare-coasttrain produce) to a Roboflow project, creating the project if
    it doesn't exist yet.

    Uses Roboflow's zip upload flow: the dataset is zipped client-side and sent
    in a single request, which is faster and more failure-resistant than
    uploading each image individually. project_type only matters when the
    project doesn't already exist; an existing project keeps its own type
    regardless of what's passed here. Waits for Roboflow to finish processing
    with its own resilient polling (see _poll_status_resilient) rather than the
    SDK's built-in wait, which has no retry on transient status-check errors.
    """
    config = load_config()
    rf = Roboflow(api_key=config["api_key"])
    workspace = rf.workspace(config.get("workspace") or None)

    with tempfile.TemporaryDirectory(prefix="roboflow_upload_") as tmp:
        upload_root = _prepare_upload_root(dataset_dir, Path(tmp))
        init_result = workspace.upload_dataset(
            dataset_path=str(upload_root),
            project_name=project_name,
            project_type=project_type,
            batch_name=batch_name,
            use_zip_upload=True,
            wait=False,
        )

    task_id = init_result["task_id"]
    print(f"Upload sent (task_id={task_id}). Waiting for Roboflow to finish processing...")
    return _poll_status_resilient(config["api_key"], workspace.url, task_id)
