"""Export the trained mini-GPT for the browser, plus a reference to check the JavaScript against."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from .generate import load_model
from .model import GPT
from .train import ARTIFACTS

REFERENCE_PROMPT = "Capitu era uma menina "
REFERENCE_NEW_TOKENS = 40


def round_to_fp16(model: GPT) -> GPT:
    """Round every parameter to float16 and back, exactly what the browser will receive."""
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(param.to(torch.float16).to(torch.float32))
    return model


def encode_fp16(tensor: torch.Tensor) -> str:
    """Little-endian float16, then base64."""
    data = tensor.detach().cpu().numpy().astype(np.float32).astype("<f2")
    return base64.b64encode(data.tobytes()).decode("ascii")


def decode_fp16(text: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Inverse of :func:`encode_fp16`."""
    raw = np.frombuffer(base64.b64decode(text), dtype="<f2").astype(np.float32)
    return torch.from_numpy(raw.reshape(shape).copy())


def tensors_to_export(model: GPT) -> dict[str, torch.Tensor]:
    """Named tensors the browser needs. The output head shares ``tok_emb`` (weight tying)."""
    return {
        name: param
        for name, param in model.named_parameters()
        if name != "head.weight"  # tied to tok_emb.weight
    }


def weights_js(model: GPT, chars: str, val_loss: float) -> str:
    """JavaScript file exposing the model as ``window.GPT_WEIGHTS``."""
    config = model.config
    payload = {
        "format": "float16-base64",
        "config": {
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
        },
        "chars": chars,
        "val_loss": round(val_loss, 4),
        "tensors": {
            name: {"shape": list(p.shape), "data": encode_fp16(p)}
            for name, p in tensors_to_export(model).items()
        },
    }
    return (
        "window.GPT_WEIGHTS = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )


def reference_json(model: GPT, chars: str) -> str:
    """Logits for a prompt and a greedy continuation, computed with the rounded model."""
    from .data import CharVocab

    vocab = CharVocab(chars)
    ids = torch.tensor([vocab.encode(REFERENCE_PROMPT)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
        last = logits[0, -1].tolist()
        out = model.generate(ids, REFERENCE_NEW_TOKENS, temperature=1e-6, top_k=1)
    return json.dumps(
        {
            "prompt": REFERENCE_PROMPT,
            "prompt_ids": ids[0].tolist(),
            "last_logits": [round(v, 5) for v in last],
            "greedy_ids": out[0].tolist(),
            "greedy_text": vocab.decode(out[0].tolist()),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def main() -> None:
    """Load the checkpoint, round to float16 and write the two files."""
    torch.set_num_threads(2)
    model, vocab = load_model()
    checkpoint = torch.load(ARTIFACTS / "gpt.pt", map_location="cpu", weights_only=True)
    val_loss = float(checkpoint["val_loss"])
    round_to_fp16(model)
    (ARTIFACTS / "gpt_weights.js").write_text(
        weights_js(model, vocab.chars, val_loss), encoding="utf-8"
    )
    (ARTIFACTS / "gpt_reference.json").write_text(
        reference_json(model, vocab.chars), encoding="utf-8"
    )
    size = Path(ARTIFACTS / "gpt_weights.js").stat().st_size
    print(f"validation loss {val_loss:.3f}; wrote gpt_weights.js ({size / 1e6:.2f} MB)")
    print("wrote gpt_reference.json")


if __name__ == "__main__":
    main()
