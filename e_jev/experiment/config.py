"""Small JSON configuration with explicit dataclass settings at module boundaries."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["variant"] not in ("base", "jev"):
        raise ValueError("variant must be base or jev")
    if config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    return config


def select_tasks(config):
    selection = json.loads((ROOT / "experiment/tasks.json").read_text())
    requested = config["tasks"]
    if requested is not None:
        if not set(requested) <= set(selection["task_ids"]):
            raise ValueError("unknown task ids")
        tasks = [task for task in selection["task_ids"] if task in requested]
    else:
        tasks = selection["task_ids"][:]
    limit = config["task_limit"]
    if limit is not None:
        if limit < 1:
            raise ValueError("task_limit must be positive or null")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError("no tasks selected")
    return selection, tasks
