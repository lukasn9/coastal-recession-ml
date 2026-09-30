import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roboflow_config import RoboflowConfigMissing
from src.roboflow_upload import check_upload_status, upload_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Upload a COCO-format dataset (from `main.py build-dataset --export-format "
        "coco` or `main.py prepare-coasttrain`) to a Roboflow project. Requires credentials set "
        "via `main.py roboflow-config` first."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Path to the dataset directory, contains train/ and val(id)/ subfolders, "
        "each with images and _annotations.coco.json",
    )
    parser.add_argument("--project", help="Roboflow project name. Created if it doesn't exist.")
    parser.add_argument(
        "--project-type",
        default="instance-segmentation",
        help="Project type, only used if --project doesn't exist yet (default: instance-segmentation)",
    )
    parser.add_argument("--batch-name", default=None, help="Batch name for this upload (default: auto-generated)")
    parser.add_argument(
        "--check-task-id",
        default=None,
        metavar="TASK_ID",
        help="Skip uploading and just check the status of a previous upload by its task_id "
        "(printed when that upload was sent). Useful if status polling failed but the upload "
        "itself, which happens before polling starts, likely went through.",
    )
    args = parser.parse_args()

    try:
        if args.check_task_id:
            result = check_upload_status(args.check_task_id)
        else:
            if not args.dataset_dir or not args.project:
                parser.error("--dataset-dir and --project are required unless --check-task-id is given")
            result = upload_dataset(
                args.dataset_dir,
                project_name=args.project,
                project_type=args.project_type,
                batch_name=args.batch_name,
            )
    except RoboflowConfigMissing as e:
        print(e)
        return

    print(f"\nDone. {result}")


if __name__ == "__main__":
    main()
