"""Serve the official Clef joint decision head in its own Python environment.

This is a local single-request server: the model is loaded once, but each record
uses an independent forward with no inference cache. No Agent code is imported.
"""

import argparse
import gc
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from classification.release import MODEL_ID, REVISION


def load_helper(path):
    name = "clef_release_joint_schema_model"
    spec = importlib.util.spec_from_file_location(name, Path(path) / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # required by dataclasses in the official helper
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/home/cache/clef"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=37753)
    parser.add_argument("--max-input-tokens", type=int, default=65536)
    args = parser.parse_args()
    root = args.cache_root.expanduser().resolve()
    for key, relative in {
        "HF_HOME": "huggingface", "HF_MODULES_CACHE": "modules",
        "HF_HUB_CACHE": "huggingface/hub", "HUGGINGFACE_HUB_CACHE": "huggingface/hub",
        "TORCH_HOME": "torch", "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "CUDA_CACHE_PATH": "cuda", "XDG_CACHE_HOME": "xdg",
    }.items():
        os.environ[key] = str(root / relative)

    import torch
    import transformers
    from huggingface_hub import snapshot_download

    path = args.model_path or Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=root / "huggingface/hub"
    ))
    helper = load_helper(path)
    model, processor = helper.load_release_model(path, device=args.device)
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = processor.tokenizer
    limit = min(args.max_input_tokens, model.language_model.config.text_config.max_position_embeddings)
    metadata = {
        "model": MODEL_ID, "release_revision": REVISION if args.model_path is None else None,
        "model_path": str(path), "custom_model_path": args.model_path is not None,
        "max_input_tokens": limit, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    def infer(record):
        started = time.perf_counter()
        # The official encoder silently truncates STATE. Encode without that
        # budget first and explicitly reject oversized inputs instead.
        encoded = helper.encode_record(tokenizer, record, max_length=2**31 - 1)
        if len(encoded.input_ids) > limit:
            raise ValueError(f"Clef input has {len(encoded.input_ids)} tokens; limit is {limit}; no truncation performed")
        batch = helper.collate_records([encoded], tokenizer.pad_token_id, torch.device(args.device))
        with torch.inference_mode():
            logits = model(batch)[0]
        answers = {
            question.question_id: dict(zip(question.option_ids, scores.float().softmax(-1).tolist()))
            for question, scores in zip(encoded.questions, logits, strict=True)
        }
        return {
            "answers": answers, "input_tokens": len(encoded.input_ids),
            "questions": len(encoded.questions), "wall_seconds": time.perf_counter() - started,
            "model": metadata["model"], "release_revision": metadata["release_revision"],
        }

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            self.reply(200, metadata) if self.path == "/health" else self.reply(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/classify":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                record = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(200, infer(record))
            except ValueError as error:
                self.reply(422, {"error": str(error)})
            except Exception as error:
                self.reply(500, {"error": f"{type(error).__name__}: {error}"})

    class ClefServer(HTTPServer):
        def finish_request(self, request, client_address):
            try:
                return super().finish_request(request, client_address)
            finally:
                # Clean up after the handler and infer() locals have been released.
                gc.collect()
                torch.cuda.synchronize(args.device)
                torch.cuda.empty_cache()

    print(json.dumps(metadata), flush=True)
    with ClefServer(("127.0.0.1", args.port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
