"""Append-only token tape; classification and environment execution live outside it."""

from dataclasses import dataclass
import ast
import json
from pathlib import Path
import time
import torch

from classification.types import TokenAction
from observation.alignment import TokenizedObservation


class Protocol:
    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).parent / "assets/agent.json"
        self.config = json.loads(path.read_text())

    def initial(self, task):
        return (
            "<|im_start|>system\n" + self.config["system_prompt"] + "<|im_end|>\n"
            "<|im_start|>user\n" + task + "<|im_end|>\n"
        )

    def assistant(self):
        return "<|im_start|>assistant\n" + self.config["assistant_prefix"]


class Tape:
    def __init__(self, tokenizer, protocol, task):
        self.tokenizer, self.protocol = tokenizer, protocol
        self.ids, self.spans, self.actions = [], [], []
        self.span_id, self.cache = 0, None
        self.append(protocol.initial(task))

    def append_ids(self, ids, *, span=-1, actions=None):
        self.ids.extend(ids)
        self.spans.extend([span] * len(ids))
        self.actions.extend([TokenAction.KEEP] * len(ids) if actions is None else actions)

    def append(self, text):
        self.append_ids(self.tokenizer.encode(text, add_special_tokens=False))

    def observation(self, encoded: TokenizedObservation):
        end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if self.ids[-len(end_ids):] != end_ids:
            self.append("<|im_end|>")
        self.append("\n<|im_start|>user\n<tool_result>\n")
        start = len(self.ids)
        self.append_ids(encoded.ids, span=self.span_id, actions=encoded.actions)
        self.append("\n</tool_result><|im_end|>\n")
        self.span_id += 1
        return start

    def uncached(self, device):
        offset = 0 if self.cache is None else self.cache.logical_length
        return {
            "ids": torch.tensor(self.ids[offset:], dtype=torch.long, device=device),
            "span": torch.tensor(self.spans[offset:], dtype=torch.long, device=device),
            "actions": torch.tensor(self.actions[offset:], dtype=torch.long, device=device),
            "cache": self.cache,
        }


@dataclass(frozen=True)
class GenerationConfig:
    max_action_tokens: int = 768
    max_context: int = 32768
    greedy: bool = True
    temperature: float = 0.6
    top_k: int = 20

    def __post_init__(self):
        if min(self.max_action_tokens, self.max_context, self.top_k) < 1 or self.temperature <= 0:
            raise ValueError("generation budgets, top_k and temperature must be positive")


def parse_action(text, protocol):
    opening, closing = protocol.config["action_open"], protocol.config["action_close"]
    if opening not in text or closing not in text.split(opening, 1)[1]:
        return None
    code = text.split(opening, 1)[1].split(closing, 1)[0].strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code or None


class Agent:
    def __init__(self, model, tokenizer, protocol, generation):
        self.model, self.tokenizer, self.protocol = model, tokenizer, protocol
        self.generation = generation

    @torch.inference_mode()
    def generate(self, tape, generator):
        tape.append(self.protocol.assistant())
        initial = len(tape.ids)
        generated, events = [], []
        forward_tokens = layer_positions = attention_pairs = 0
        started, ttft, stop = time.perf_counter(), None, "action_token_limit"
        for _ in range(self.generation.max_action_tokens):
            if len(tape.ids) >= self.generation.max_context:
                stop = "context_limit"
                break
            kwargs = tape.uncached(self.model.device)
            logits, output = self.model.next_logits(**kwargs)  # 模型前向
            tape.cache = output.cache
            events.extend(output.events)
            forward_tokens += len(kwargs["ids"])
            layer_positions += sum(output.layer_lengths)
            attention_pairs += output.attention_pairs
            if self.generation.greedy:
                token = int(logits.argmax())
            else:
                values, indices = (logits.float() / self.generation.temperature).topk(self.generation.top_k)
                token = int(indices[torch.multinomial(values.softmax(-1), 1, generator=generator)])
            tape.append_ids([token])
            generated.append(token)
            if ttft is None:
                ttft = time.perf_counter() - started
            text = self.tokenizer.decode(generated, skip_special_tokens=False)
            if parse_action(text, self.protocol) is not None:
                stop = "action_complete"
                break
            if token in (self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")):
                stop = "eos_without_complete_action"
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return parse_action(text, self.protocol), {
            "text": text, "stop": stop, "generated_ids": generated,
            "input_tokens": initial, "output_tokens": len(generated),
            "forward_input_tokens": forward_tokens, "token_layer_positions": layer_positions,
            "attention_pairs": attention_pairs, "compression_events": events,
            "wall_seconds": time.perf_counter() - started, "ttft_seconds": ttft,
            "cache_bytes": tape.cache.bytes() if tape.cache else 0,
            "cache_layer_lengths": list(tape.cache.layer_lengths) if tape.cache else [],
        }
