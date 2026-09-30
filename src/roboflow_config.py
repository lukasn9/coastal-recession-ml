import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / ".roboflow"
CONFIG_PATH = CONFIG_DIR / "config.json"

_TEMPLATE = {"api_key": "", "workspace": ""}


class RoboflowConfigMissing(RuntimeError):
    """Raised when .roboflow/config.json doesn't exist yet or has no api_key set."""


def write_config(config: dict) -> None:
    """Write the Roboflow config, creating .roboflow/ if it doesn't exist yet."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")


def load_config() -> dict:
    """
    Load Roboflow credentials from .roboflow/config.json. If the file doesn't
    exist yet, creates it with an empty template and raises, so the caller gets
    a clear one-time message telling them where to fill in their API key,
    rather than a confusing authentication error from deeper in the SDK.
    """
    if not CONFIG_PATH.exists():
        write_config(_TEMPLATE)
        raise RoboflowConfigMissing(
            f"Created {CONFIG_PATH} with an empty template. Fill in your Roboflow "
            "API key (and workspace, if you have more than one) and run this again."
        )

    config = json.loads(CONFIG_PATH.read_text())
    if not config.get("api_key"):
        raise RoboflowConfigMissing(f"{CONFIG_PATH} exists but has no api_key set. Fill it in and run this again.")
    return config


def set_credentials(api_key: str | None = None, workspace: str | None = None) -> dict:
    """Update stored credentials, keeping whatever isn't passed unchanged. Creates the config if missing."""
    config = dict(_TEMPLATE)
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text()))
    if api_key is not None:
        config["api_key"] = api_key
    if workspace is not None:
        config["workspace"] = workspace
    write_config(config)
    return config
