"""AppWorld 接入；不依赖角色、模型或 runner 版本。"""

from contextlib import contextmanager
import os
from urllib.parse import urlparse


def check_environment_service(environment_url: str) -> None:
    """本机环境不走代理；在加载模型前检查服务是否可达。"""
    import requests

    if urlparse(environment_url).hostname in {"127.0.0.1", "localhost", "::1"}:
        for key in ("NO_PROXY", "no_proxy"):
            entries = [item.strip() for item in os.environ.get(key, "").split(",") if item.strip()]
            for host in ("localhost", "127.0.0.1", "::1"):
                if host not in entries:
                    entries.append(host)
            os.environ[key] = ",".join(entries)
    try:
        response = requests.get(environment_url.rstrip("/") + "/", timeout=5)
        response.raise_for_status()
        if "AppWorld" not in response.json().get("message", ""):
            raise ValueError("response is not from an AppWorld environment server")
    except (requests.RequestException, ValueError, AttributeError) as error:
        raise RuntimeError(
            f"AppWorld service check failed at {environment_url}: {error}. "
            "Start tools/serve_environment.py first and check its terminal logs."
        ) from error


def prepare_environment(root: str, expected_task_ids: list[str]) -> None:
    from appworld import load_task_ids, update_root

    update_root(root)
    if set(load_task_ids("test_normal")) != set(expected_task_ids):
        raise ValueError("AppWorld test_normal differs from the recorded task selection")


@contextmanager
def open_environment(task_id: str, experiment_name: str, environment_url: str):
    from appworld import AppWorld

    with AppWorld(
        task_id=task_id, experiment_name=experiment_name,
        remote_environment_url=environment_url, remote_docker=True,
    ) as world:
        if task_identifier(world) != task_id:
            raise ValueError("AppWorld opened an unexpected task")
        yield world


def task_identifier(world) -> str:
    task_id = getattr(world.task, "id", None)
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("AppWorld Task.id must be a nonempty string")
    if getattr(world, "task_id", task_id) != task_id:
        raise ValueError("AppWorld.task_id differs from Task.id")
    return task_id


def task_instruction(task) -> str:
    supervisor = task.supervisor
    if not isinstance(supervisor, dict):
        return "Task: " + str(task.instruction)
    name = " ".join(str(supervisor.get(k, "")).strip() for k in ("first_name", "last_name")).strip()
    facts = [f"My name is {name}."] if name else []
    for field, label in (("email", "personal email"), ("phone_number", "phone number")):
        if supervisor.get(field):
            facts.append(f"My {label} is {supervisor[field]}.")
    return (" ".join(facts) + "\n\n" if facts else "") + "Task: " + str(task.instruction)


def execute_action(world, action: str) -> str:
    # AppWorld 的 Python 变量跨轮保留；执行返回的错误文本属于 observation。
    # 网络断开等基础设施异常不伪装为普通任务失败。
    return str(world.execute(action))


def task_completed(world) -> bool:
    return bool(world.task_completed())


def official_score(world, split: str) -> dict:
    tracker = world.evaluate()
    score = {
        "success": bool(tracker.success), "passed": int(tracker.pass_count),
        "total": int(tracker.num_tests), "source": "appworld.world.evaluate", "split": split,
    }
    validate_score(score, split)
    return score


def validate_score(score: dict, split: str) -> None:
    if (
        score.get("split") != split or type(score.get("success")) is not bool
        or type(score.get("passed")) is not int or type(score.get("total")) is not int
        or not 0 <= score["passed"] <= score["total"] or score["total"] == 0
        or score["success"] != (score["passed"] == score["total"])
        or score.get("source") not in (
            "appworld.world.evaluate", "appworld.evaluator.evaluate_task"
        )
    ):
        raise ValueError("invalid official evaluator evidence")
