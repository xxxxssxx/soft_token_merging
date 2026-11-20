"""Atomic records and summaries; no model or benchmark execution."""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from experiment.config import ROOT


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def new_run_directory(requested):
    run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    path = Path(requested).expanduser().resolve() if requested else ROOT / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path, run_id


def write_json(path, value):
    if "turns" in value:
        first = (
            "completed", "official_score", "instruction", "termination_reason",
            "task_wall_seconds", "finished_at",
        )
        value = {**{key: value[key] for key in first if key in value}, **value}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def task_summary(record):
    agent = [turn["agent"] for turn in record["turns"] if "agent" in turn]
    classifiers = [turn["classification"] for turn in record["turns"] if "classification" in turn]
    events = [turn["compression"] for turn in record["turns"] if turn.get("compression") is not None]
    before = sum(event["tokens_before"] for event in events)
    after = sum(event["tokens_after"] for event in events)
    return {
        "task_id": record["task_id"], "official_score": record["official_score"],
        "completed": record["completed"], "termination_reason": record["termination_reason"],
        "task_wall_seconds": record["task_wall_seconds"],
        "actions": sum("raw_observation" in turn for turn in record["turns"]),
        "agent": {key: sum(item[key] for item in agent) for key in (
            "input_tokens", "output_tokens", "forward_input_tokens", "token_layer_positions",
            "attention_pairs", "wall_seconds",
        )},
        "clef": {
            "calls": sum(len(item.get("calls", [])) for item in classifiers),
            "input_tokens": sum(item.get("input_tokens", 0) for item in classifiers),
            "output_tokens": 0,
            "wall_seconds": sum(item.get("wall_seconds", 0) for item in classifiers),
        },
        "compression": {
            "processed_observations": len(events),
            "unprocessed_observations": sum(
                "classification" in turn and turn.get("compression") is None for turn in record["turns"]
            ),
            "tokens_before": before, "tokens_after": after,
            "retained_fraction": after / before if before else None,
            **{key: sum(item[key] for item in events) for key in (
                "merge_requested", "merged_tokens", "soft_tokens", "deleted_tokens",
            )},
        },
    }


def summarize(rows):
    return {
        "tasks": len(rows),
        "successes": sum(row["official_score"]["success"] for row in rows),
        "task_results": rows,
        "cost_notes": (
            "Agent attention_pairs counts physical causal query-key pairs in full-attention layers, "
            "including prefill and decode. Clef forwards are reported separately and are not included "
            "in that count. Logical input tokens are not physical cache lengths. "
            "Final observations without a subsequent Agent forward have no compression result."
        ),
    }
