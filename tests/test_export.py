from __future__ import annotations

import json

import torch

from mini_gpt.export import (
    decode_fp16,
    encode_fp16,
    reference_json,
    round_to_fp16,
    tensors_to_export,
    weights_js,
)
from mini_gpt.model import GPT, GPTConfig

CHARS = " abcdefghijklmnopqrstuvwxyzC"


def tiny_model() -> GPT:
    torch.manual_seed(0)
    return GPT(
        GPTConfig(vocab_size=len(CHARS), block_size=64, n_layer=2, n_head=2, n_embd=16)
    ).eval()


def test_fp16_roundtrip_is_close() -> None:
    values = torch.randn(6, 5)
    decoded = decode_fp16(encode_fp16(values), (6, 5))
    assert torch.allclose(decoded, values, atol=2e-3)


def test_round_to_fp16_keeps_weight_tying() -> None:
    model = round_to_fp16(tiny_model())
    assert model.head.weight is model.tok_emb.weight
    values = model.tok_emb.weight.detach()
    assert torch.equal(values, values.to(torch.float16).to(torch.float32))


def test_exported_tensors_skip_the_tied_head() -> None:
    names = set(tensors_to_export(tiny_model()))
    assert "head.weight" not in names
    assert {"tok_emb.weight", "pos_emb.weight", "ln_f.weight", "blocks.1.mlp.2.bias"} <= names


def test_weights_js_is_valid_and_complete() -> None:
    model = round_to_fp16(tiny_model())
    text = weights_js(model, CHARS, 1.234)
    assert text.startswith("window.GPT_WEIGHTS = ") and text.endswith(";\n")
    data = json.loads(text[len("window.GPT_WEIGHTS = ") : -2])
    assert data["chars"] == CHARS and data["config"]["n_layer"] == 2
    tensor = data["tensors"]["blocks.0.attn.qkv.weight"]
    rebuilt = decode_fp16(tensor["data"], tuple(tensor["shape"]))
    assert torch.equal(rebuilt, model.get_parameter("blocks.0.attn.qkv.weight").detach())


def test_reference_is_deterministic_and_complete() -> None:
    model = round_to_fp16(tiny_model())
    first = json.loads(reference_json(model, CHARS))
    second = json.loads(reference_json(model, CHARS))
    assert first == second
    assert len(first["last_logits"]) == len(CHARS)
    assert len(first["greedy_ids"]) == len(first["prompt_ids"]) + 40
