"""Exercise the existing HTTP retry library against a loopback server only."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

import pytest
import requests

import classification.jev as jev
from classification.types import ClassificationRequest
from observation.fields import split_observation


@pytest.mark.parametrize("recover", [True, False])
def test_bounded_retries_and_rate_limit_handling(monkeypatch, recover):
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            attempts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            success = recover and len(attempts) == 3
            body = {
                "model": "jev-1.14.0", "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {"field_0": {"probabilities": {"relevant": 0.8, "irrelevant": 0.2}}},
            } if success else {"detail": "rate limited" if recover else "overloaded"}
            self.send_response(200 if success else 429 if recover else 529)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-jev-key")
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setattr(jev, "API_URL", f"http://127.0.0.1:{server.server_port}/v1/systemone")
        classifier = jev.JevClassifier(jev.JevConfig())
        classifier.session.trust_env = False
        classifier.session.mount("http://", classifier.session.get_adapter("https://"))
        request = ClassificationRequest("task", [], "action", split_observation("1"), "prompt")
        try:
            if recover:
                result = classifier.classify(request)
                assert result.metadata["input_tokens"] == 10
                assert len(result.metadata["calls"]) == 1
            else:
                with pytest.raises(requests.HTTPError):
                    classifier.classify(request)
            assert len(attempts) == 3
        finally:
            classifier.close()
            server.shutdown()
            thread.join()
