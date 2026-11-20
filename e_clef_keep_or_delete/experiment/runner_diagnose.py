"""Diagnostic runner: optionally keep three tasks' suspect fields before token alignment."""

import argparse
import ast
from dataclasses import replace
from importlib.metadata import version
import json
import time
import torch

from classification.clef import ClefClassifier, ClefConfig
from classification.types import ClassificationRequest, ClassificationResult, FieldDecision, TokenAction
from compression.operator import CompressionConfig
from experiment.config import ROOT, load_config, select_tasks
from experiment.environment import (
    check_environment_service, execute_action, official_score, open_environment,
    prepare_environment, task_completed, task_instruction,
)
from experiment.recording import new_run_directory, summarize, task_summary, timestamp, write_json
from observation.alignment import align_observation
from observation.fields import split_observation
from runtime.agent import Agent, GenerationConfig, Protocol, Tape
from runtime.model import load_model



def apply_keep_override(task_id, action, segmented, classified):
    """Diagnostic intervention only; preserve Clef labels and probabilities."""
    calls = [node for node in ast.walk(ast.parse(action)) if isinstance(node, ast.Call)]

    def called(app, method, **keywords):
        for call in calls:
            if ast.unparse(call.func) != f"apis.{app}.{method}":
                continue
            values = {kw.arg: kw.value.value for kw in call.keywords
                      if isinstance(kw.value, ast.Constant)}
            if all(values.get(key) == value for key, value in keywords.items()):
                return True
        return False

    targets, reason = set(), None
    if task_id == "fd1f8fa_1" and called("spotify", "show_liked_songs"):
        targets = {field.id for field in segmented.fields}
        reason = "Liked-song records are needed to match songs in the queue."
    elif task_id == "21abae1_3" and called(
        "api_docs", "show_api_doc", app_name="venmo", api_name="login"
    ):
        targets = {field.id for field in segmented.fields
                   if field.path == ("response_schemas", "success", "access_token")}
        reason = "The login response schema is needed to extract the access token."
    elif task_id == "29a7b7e_2" and segmented.format == "json" and called(
        "api_docs", "show_api_descriptions", app_name="file_system"
    ):
        for field in segmented.fields:
            value = json.loads(field.value_text)
            if isinstance(value, dict) and value.get("name") == "login":
                targets.add(field.id)
        reason = "The login API is a prerequisite for accessing the file system."

    paths = {field.id: list(field.path) for field in segmented.fields}
    overrides = [{
        "field_id": decision.field_id, "path": paths[decision.field_id],
        "original_label": decision.label, "original_action": decision.action.name,
        "probabilities": decision.probabilities, "executed_action": "KEEP",
        "changed": decision.action != TokenAction.KEEP, "reason": reason,
    } for decision in classified.decisions if decision.field_id in targets]
    effective = ClassificationResult([
        replace(decision, action=TokenAction.KEEP) if decision.field_id in targets else decision
        for decision in classified.decisions
    ], classified.metadata)
    return effective, overrides


def collect_task(world, agent, classifier, config, record, save):
    """Only this loop advances history; the classifier sees completed prior turns."""
    instruction = task_instruction(world.task)
    record.update(instruction=instruction, turns=[], phase="running")
    tape = Tape(agent.tokenizer, agent.protocol, instruction)
    generator = torch.Generator(device=agent.model.device).manual_seed(record["seed"])
    history, reason = [], "step_limit"
    save()
    for step in range(config["max_steps"]):
        turn = {"step": step + 1, "phase": "generating", "compression": None}
        record["turns"].append(turn)
        save()
        action, generation = agent.generate(tape, generator)  # next token循环
        generation = {key: value for key, value in generation.items()
                      if key not in ("generated_ids", "cache_layer_lengths")}
        generation["compression_events"] = [
            {key: value for key, value in event.items() if key != "soft_groups"}
            for event in generation["compression_events"]
        ]
        turn.update(action=action, agent=generation, phase="generated")
        # Compression belongs to the observation that produced these positions,
        # even though it happens during the following turn's Agent prefill.
        for event in generation["compression_events"]:
            record["turns"][event["span"]]["compression"] = event
        save()
        if action is None:
            reason = generation["stop"]
            break

        turn["phase"] = "executing"
        save()
        raw = execute_action(world, action)  # 与环境交互，产生 observation
        turn.update(raw_observation=raw, phase="observed")
        save()
        segmented = split_observation(raw)
        turn["fields"] = [
            {"id": field.id, "path": list(field.path), "text": field.text}
            for field in segmented.fields
        ]
        turn["observation_format"] = segmented.format
        turn["phase"] = "classifying"
        save()
        # 对 observation 分类
        if classifier is None:
            classified = ClassificationResult([
                FieldDecision(field.id, TokenAction.KEEP, "base") for field in segmented.fields
            ], {"method": "base", "calls": [], "input_tokens": 0, "wall_seconds": 0.0})  # 不分类基线
        else:
            classified = classifier.classify(ClassificationRequest(
                task=instruction, history=history, action=action, observation=segmented,
                agent_system_prompt=agent.protocol.config["system_prompt"],
            ))
        original_actions = {decision.field_id: decision.action.name for decision in classified.decisions}
        enabled = config.get("diagnostic_keep_override", False) and classifier is not None
        overrides = []
        if enabled:
            classified, overrides = apply_keep_override(record["task_id"], action, segmented, classified)
        turn["diagnostic_keep_override"] = {"enabled": enabled, "overrides": overrides}
        encoded = align_observation(segmented, classified.decisions, agent.tokenizer)

        tape.observation(encoded)  # 原始 token 和动作追加到 Tape，下一次前向执行压缩。
        decisions = {decision.field_id: decision for decision in classified.decisions}
        turn.update(
            phase="appended", classification=classified.metadata,
            fields=[{
                **field,
                "label": decisions[field["id"]].label,
                "action": decisions[field["id"]].action.name,
                "classifier_action": original_actions[field["id"]],
                "probabilities": decisions[field["id"]].probabilities,
            } for field in turn["fields"]],
        )
        history.append({"action": action, "assistant": generation["text"], "observation": raw})
        save()
        print(json.dumps({"task": record["task_id"], "step": step + 1,
                          "fields": len(segmented.fields), "observation_token_count": len(encoded.ids)}), flush=True)
        if task_completed(world):
            reason = "completed"
            break
        if len(tape.ids) >= agent.generation.max_context:
            reason = "context_limit"
            break

    record.update(completed=task_completed(world), termination_reason=reason, phase="evaluating")
    save()
    record["official_score"] = official_score(world, "test_normal")
    record["phase"] = "finished"
    save()


def run(config):
    from transformers import AutoTokenizer

    selection, tasks = select_tasks(config)
    compression = CompressionConfig(**config["compression"])
    generation = GenerationConfig(**config["generation"])
    protocol = Protocol()
    output, run_id = new_run_directory(config["output_directory"])
    classifier = ClefClassifier(ClefConfig(**config["clef"])) if config["variant"] == "clef" else None
    write_json(output / "run.json", {
        "run_id": run_id, "started_at": timestamp(), "config": config, "tasks": tasks,
        "runner": "runner_diagnose", "diagnostic_policy": "keep_suspect_fields_v1",
        "diagnostic_keep_override": config.get("diagnostic_keep_override", False),
        "task_seeds": {task: selection["seed"] + selection["task_ids"].index(task) for task in tasks},
        "agent_prompt": protocol.config, "classification_prompt": classifier.prompt if classifier else None,
    })
    print(f"Run directory: {output}", flush=True)
    rows, record, trajectory = [], None, None
    try:
        write_json(output / "status.json", {"phase": "loading", "completed": 0, "total": len(tasks)})
        service = classifier.health() if classifier else None
        prepare_environment(config["appworld_root"], selection["task_ids"])
        check_environment_service(config["environment_url"])
        torch.manual_seed(config["model_seed"])
        model = load_model(config["model"]["path"], compression,
                           config["model"]["device"], config["model"]["dtype"])  # 模型实例。压缩写在模型的forward。
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], local_files_only=True)
        agent = Agent(model, tokenizer, protocol, generation)
        write_json(output / "runtime.json", {
            "agent_packages": {name: version(name) for name in ("torch", "transformers", "appworld")},
            "clef_service": service, "agent_device": str(model.device),
        })
        for task_id in tasks:
            trajectory = output / "trajectories" / f"test_normal-{task_id}.json"
            record = {
                "task_id": task_id, "run_id": run_id, "variant": config["variant"],
                "seed": selection["seed"] + selection["task_ids"].index(task_id),
                "phase": "opening_environment", "started_at": timestamp(),
                "turns": [], "completed": False, "official_score": None,
            }
            save = lambda: write_json(trajectory, record)
            save()
            started = time.perf_counter()
            with open_environment(task_id, f"e2-{run_id}-{task_id}", config["environment_url"]) as world:
                collect_task(world, agent, classifier, config, record, save)  # 任务循环
            record.update(task_wall_seconds=time.perf_counter() - started, finished_at=timestamp())
            save()
            rows.append(task_summary(record))
            write_json(output / "summary.json", summarize(rows))
            write_json(output / "status.json", {
                "phase": "running", "completed": len(rows), "total": len(tasks), "last_task": task_id,
            })
        write_json(output / "status.json", {"phase": "finished", "completed": len(rows), "total": len(tasks)})
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        if record is not None:
            record["error"] = failure
            write_json(trajectory, record)
        write_json(output / "status.json", {"phase": "error", "completed": len(rows), "error": failure})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "experiment/config.json")
    parser.add_argument("--variant", choices=("base", "clef"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--output")
    parser.add_argument(
        "--diagnostic-keep-override", action=argparse.BooleanOptionalAction, default=None,
        help="Keep the suspect fields of fd1f8fa_1, 21abae1_3 and 29a7b7e_2 (default: off).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for argument, key in ((args.variant, "variant"), (args.limit, "task_limit"),
                          (args.tasks, "tasks"), (args.output, "output_directory")):
        if argument is not None:
            config[key] = argument
    if args.diagnostic_keep_override is not None:
        config["diagnostic_keep_override"] = args.diagnostic_keep_override
    run(config)


if __name__ == "__main__":
    main()
