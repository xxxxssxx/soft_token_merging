"""Prepared HTTP requests are intercepted here; tests never call the Jev API."""

import json
from types import SimpleNamespace

import pytest
import requests


@pytest.fixture
def jev_api(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-jev-key")
    api = SimpleNamespace(session=requests.Session(), requests=[], status=200, body=None)

    def send(request, **kwargs):
        api.requests.append((request, kwargs))
        record = json.loads(request.body)
        body = api.body if api.body is not None else {
            "model": "jev-1.14.0",
            "answers": {
                key: {
                    "type": "choice", "choice": "relevant" if key == "field_0" else "irrelevant",
                    "confidence": 0.9,
                    "probabilities": {"relevant": 0.9 if key == "field_0" else 0.1,
                                      "irrelevant": 0.1 if key == "field_0" else 0.9},
                } for key in record["questions"]
            },
            "usage": {"input_tokens": 100, "output_tokens": 12},
        }
        response = requests.Response()
        response.status_code = api.status
        response.url, response.request = request.url, request
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(api.session, "send", send)
    return api
