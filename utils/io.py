import json
import hashlib
from pathlib import Path


def generate_sim_id(config: dict) -> str:
    """Generate a short unique ID string from a configuration dict."""
    json_str = json.dumps(config, sort_keys=True)
    return hashlib.sha1(json_str.encode("utf-8")).hexdigest()[:8]


def write_metadata(path: str | Path, data: dict) -> None:
    """Write dictionary to JSON file located at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_metadata(path: str | Path) -> dict:
    """Read JSON file located at *path* and return the dictionary."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
