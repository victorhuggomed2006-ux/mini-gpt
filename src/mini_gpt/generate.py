"""Load a trained checkpoint and continue a piece of text."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import CharVocab
from .model import GPT, GPTConfig
from .train import ARTIFACTS, CHECKPOINT

WEIGHTS_ONLY = ARTIFACTS / "gpt_model.pt"


def load_model(path: Path | None = None) -> tuple[GPT, CharVocab]:
    """Rebuild the model and vocabulary from a checkpoint (or from the published weights)."""
    if path is None:
        path = CHECKPOINT if CHECKPOINT.exists() else WEIGHTS_ONLY
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model.eval(), CharVocab(state["chars"])


def complete(
    model: GPT,
    vocab: CharVocab,
    prompt: str,
    length: int = 300,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int | None = None,
) -> str:
    """Return ``prompt`` followed by ``length`` generated characters."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = torch.tensor([vocab.encode(prompt) or [0]], dtype=torch.long)
    out = model.generate(ids, length, temperature=temperature, top_k=top_k)
    return vocab.decode(out[0].tolist())


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Capitu ")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    model, vocab = load_model()
    print(complete(model, vocab, args.prompt, args.length, args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
