import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.roboflow_config import CONFIG_PATH, set_credentials


def main():
    parser = argparse.ArgumentParser(
        description="Set stored Roboflow credentials, used by `main.py upload-roboflow`. "
        "Stored in .roboflow/config.json, which is gitignored and never committed."
    )
    parser.add_argument("--api-key", default=None, help="Roboflow API key")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Roboflow workspace name. Optional, defaults to the workspace tied to your API key.",
    )
    args = parser.parse_args()

    if args.api_key is None and args.workspace is None:
        parser.error("pass at least one of --api-key or --workspace")

    set_credentials(api_key=args.api_key, workspace=args.workspace)
    print(f"Saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
