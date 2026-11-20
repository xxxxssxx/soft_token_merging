"""Field relevance client. The Agent process never imports Clef's Torch stack."""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from classification.types import (
    ClassificationRequest, ClassificationResult, FieldDecision, TokenAction,
)


@dataclass(frozen=True)
class ClefConfig:
    url: str = "http://127.0.0.1:37753"
    fields_per_request: int = 32
    relevance_threshold: float = 0.5
    timeout: float = 300
    prompt_version: str = "task-relevance-v3-delete"

    def __post_init__(self):
        if self.fields_per_request < 1 or not 0 <= self.relevance_threshold <= 1:
            raise ValueError("invalid Clef question budget or relevance threshold")


class ClefClassifier:
    def __init__(self, config: ClefConfig, transport=None):
        self.config = config
        prompts = json.loads(Path(__file__).with_name("prompt.json").read_text())
        self.prompt = prompts[config.prompt_version]
        self.transport = transport or self._http
        self.opener = build_opener(ProxyHandler({}))  # local services bypass shell proxies

    def _http(self, record):
        request = Request(
            self.config.url.rstrip("/") + "/classify",
            data=json.dumps(record, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.config.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"Clef HTTP {error.code}: {error.read().decode()}") from error

    def health(self):
        with self.opener.open(self.config.url.rstrip("/") + "/health", timeout=5) as response:
            return json.load(response)

    def classify(self, request: ClassificationRequest) -> ClassificationResult:
        started = time.perf_counter()
        fields = request.observation.fields
        # Clean leaf values are for classification only. Alignment still uses
        # the unmodified observation and its complete source partition.
        state = {
            "classification_policy": self.prompt["instructions"],
            "agent_context": {
                "notice": self.prompt["agent_prompt_notice"],
                "system_prompt": request.agent_system_prompt,
            },
            "task": request.task, "history": request.history,
            "current_action": request.action,
            "observation_format": request.observation.format,
            "observation_fields": [
                {"id": f.id, "path": list(f.path), "text": f.value_text} for f in fields
            ],
        }
        decisions, calls = [], []
        for start in range(0, len(fields), self.config.fields_per_request):
            batch = fields[start:start + self.config.fields_per_request]
            questions = {
                f"field_{f.id}": {
                    "type": "choice",
                    "instructions": (
                        self.prompt["question"]
                        + f"\nPath: {json.dumps(f.path, ensure_ascii=False)}"
                        + f"\nField content:\n{f.value_text}"
                    ),
                    "criteria": self.prompt["criteria"],
                }
                for f in batch
            }
            response = self.transport({"state": state, "questions": questions})
            if set(response["answers"]) != set(questions):
                raise ValueError("Clef response does not match the requested fields")
            for f in batch:
                probabilities = response["answers"][f"field_{f.id}"]
                relevant = probabilities["relevant"] >= self.config.relevance_threshold
                decisions.append(FieldDecision(
                    f.id, TokenAction.KEEP if relevant else TokenAction.DELETE,
                    "relevant" if relevant else "irrelevant", probabilities,
                ))
            calls.append({key: value for key, value in response.items() if key != "answers"})
        return ClassificationResult(decisions, {
            "method": "clef_relevance", "prompt_version": self.prompt["version"],
            "relevance_threshold": self.config.relevance_threshold,
            "calls": calls,
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": 0,
            "wall_seconds": time.perf_counter() - started,
        })
