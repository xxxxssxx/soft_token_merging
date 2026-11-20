"""Jev field relevance via the official API; no local decision-model runtime."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from classification.types import (
    ClassificationRequest, ClassificationResult, FieldDecision, TokenAction,
)


API_URL = "https://api.typesafe.ai/v1/systemone"


@dataclass(frozen=True)
class JevConfig:
    model: str = "jev-1.14.0"
    fields_per_request: int = 32
    relevance_threshold: float = 0.5
    timeout: float = 300
    max_retries: int = 2
    long_context: bool = True
    prompt_version: str = "task-relevance-v3"

    def __post_init__(self):
        if self.fields_per_request < 1 or not 0 <= self.relevance_threshold <= 1:
            raise ValueError("invalid Jev question budget or relevance threshold")


class JevClassifier:
    def __init__(self, config: JevConfig, session=None):
        self.config = config
        prompts = json.loads(Path(__file__).with_name("prompt.json").read_text())
        self.prompt = prompts[config.prompt_version]
        api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Set TYPESAFE_API_KEY to use the Jev classifier")
        self.session = session if session is not None else requests.Session()
        retry = Retry(
            total=config.max_retries, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504, 529),
            allowed_methods={"POST"}, respect_retry_after_header=True,
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
        })
        if config.long_context:
            self.session.headers["TypeSafe-Danger-Zone-Long-Context"] = "true"
        self.runtime_metadata = {
            "provider": "typesafe", "client": "requests",
            "client_version": requests.__version__, **asdict(config),
        }

    def _request(self, record):
        response = self.session.post(API_URL, json=record, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.session.close()

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
            call_started = time.perf_counter()
            response = self._request({"model": self.config.model, "state": state, "questions": questions})
            if set(response["answers"]) != set(questions):
                raise ValueError("Jev response does not match the requested fields")
            for f in batch:
                probabilities = response["answers"][f"field_{f.id}"]["probabilities"]
                relevant = probabilities["relevant"] >= self.config.relevance_threshold
                decisions.append(FieldDecision(
                    f.id, TokenAction.KEEP if relevant else TokenAction.MERGE,
                    "relevant" if relevant else "irrelevant", probabilities,
                ))
            calls.append({
                "model": response["model"], "questions": len(questions),
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": response["usage"]["output_tokens"],
                "wall_seconds": time.perf_counter() - call_started,
            })
        return ClassificationResult(decisions, {
            "method": "jev_relevance", "prompt_version": self.prompt["version"],
            "relevance_threshold": self.config.relevance_threshold,
            "calls": calls,
            "input_tokens": sum(call["input_tokens"] for call in calls),
            "output_tokens": sum(call["output_tokens"] for call in calls),
            "wall_seconds": time.perf_counter() - started,
        })
