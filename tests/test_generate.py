from __future__ import annotations

from pathlib import Path

import torch

from mini_gpt.generate import complete, load_model
from mini_gpt.model import GPT, GPTConfig

CHARS = " abcdefghijklmnopqrstuvwxyz"


def save_weights_only(path: Path) -> GPT:
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=len(CHARS), block_size=32, n_layer=2, n_head=2, n_embd=16)
    model = GPT(config)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "dropout": config.dropout,
            },
            "chars": CHARS,
        },
        path,
    )
    return model


def test_load_model_from_weights_only_file(tmp_path: Path) -> None:
    original = save_weights_only(tmp_path / "gpt_model.pt")
    model, vocab = load_model(tmp_path / "gpt_model.pt")
    assert vocab.chars == CHARS
    x = torch.randint(0, len(CHARS), (1, 8))
    original.eval()
    assert torch.allclose(model(x)[0], original(x)[0], atol=1e-6)


def test_complete_returns_prompt_plus_new_characters(tmp_path: Path) -> None:
    save_weights_only(tmp_path / "gpt_model.pt")
    model, vocab = load_model(tmp_path / "gpt_model.pt")
    text = complete(model, vocab, "abc ", length=40, seed=3)
    assert text.startswith("abc ") and len(text) == 44
    assert text == complete(model, vocab, "abc ", length=40, seed=3)  # same seed, same text
