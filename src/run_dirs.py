from pathlib import Path


def next_run_dir(base_dir: Path, prefix: str = "run") -> Path:
    """
    Pick and create the next numbered run directory under base_dir, e.g.
    prefix_1, prefix_2, matching how Ultralytics numbers its own runs/train
    folders, so repeated runs of a command never overwrite older output and
    the latest one is easy to spot.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{prefix}_"):
            suffix = p.name[len(prefix) + 1:]
            if suffix.isdigit():
                existing.append(int(suffix))

    run_dir = base_dir / f"{prefix}_{max(existing, default=0) + 1}"
    run_dir.mkdir(parents=True)
    return run_dir
