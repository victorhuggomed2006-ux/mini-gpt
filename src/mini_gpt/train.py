"""Train the mini-GPT on CPU without freezing the machine.

Gentle by design: idle process priority, 2 threads, a checkpoint every ``eval_every`` steps
(so training can be stopped and resumed), and an optional ``PAUSE`` file that makes it wait.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch

from .data import CharVocab, build_corpus, split_corpus
from .model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "gpt.pt"
PAUSE_FILE = ROOT / "PAUSE"
IDLE_PRIORITY_CLASS = 0x00000040  # Windows


def lower_priority() -> None:
    """Make this process yield to everything else (calls, browser, editor)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declare the types: without them the 64-bit process handle is truncated and the call fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), IDLE_PRIORITY_CLASS):
            raise OSError(ctypes.get_last_error(), "could not lower the process priority")
    else:
        os.nice(10)


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (input, target) windows; the target is the input shifted by one character."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts.tolist()])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT, data: torch.Tensor, batch_size: int, batches: int, generator: torch.Generator
) -> float:
    """Average loss over a few random batches."""
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, model.config.block_size, generator)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("loss is missing")
        total += float(loss.item())
    model.train()
    return total / batches


def lr_at(step: int, total: int, base_lr: float, warmup: int = 100) -> float:
    """Linear warm-up, then cosine decay down to 10% of ``base_lr``."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def fit(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    steps: int,
    batch_size: int = 32,
    base_lr: float = 3e-3,
    eval_every: int = 250,
    seed: int = 0,
    start_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    on_checkpoint: Callable[[int, float, torch.optim.Optimizer], None] | None = None,
    max_seconds: float | None = None,
    pause_file: Path | None = None,
) -> float:
    """Train ``model`` from ``start_step`` to ``steps``. Returns the last validation loss."""
    generator = torch.Generator().manual_seed(seed + start_step)
    opt = optimizer or torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    started = time.time()
    val_loss = float("nan")
    model.train()
    for step in range(start_step, steps):
        while pause_file is not None and pause_file.exists():
            time.sleep(2)  # paused (for example, during a call)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, base_lr)
        x, y = get_batch(train_data, batch_size, model.config.block_size, generator)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("loss is missing")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = step + 1 == steps
        timed_out = max_seconds is not None and time.time() - started > max_seconds
        if (step + 1) % eval_every == 0 or last or timed_out:
            val_loss = estimate_loss(model, val_data, batch_size, 20, generator)
            elapsed = time.time() - started
            print(
                f"step {step + 1:5d}/{steps} train={loss.item():.3f} val={val_loss:.3f} "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
            if on_checkpoint is not None:
                on_checkpoint(step + 1, val_loss, opt)
        if timed_out:
            print("time limit reached: checkpoint saved, run again with --resume")
            break
    return val_loss


def save_checkpoint(
    model: GPT, vocab: CharVocab, step: int, val_loss: float, opt: torch.optim.Optimizer
) -> None:
    """Write model, optimizer and vocabulary so training can resume."""
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": asdict(model.config),
            "chars": vocab.chars,
        },
        CHECKPOINT,
    )


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--minutes", type=float, default=None, help="stop after this long")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lower_priority()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    corpus = build_corpus(RAW_DIR)
    vocab, train_data, val_data = split_corpus(corpus)
    print(f"corpus: {len(corpus):,} characters, vocabulary: {vocab.size}")

    start = 0
    config = GPTConfig(vocab_size=vocab.size)
    model = GPT(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.resume and CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start = int(state["step"])
        print(f"resuming from step {start}")
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params:,}")

    fit(
        model,
        train_data,
        val_data,
        steps=args.steps,
        batch_size=args.batch_size,
        base_lr=args.lr,
        start_step=start,
        optimizer=opt,
        on_checkpoint=lambda step, loss, o: save_checkpoint(model, vocab, step, loss, o),
        max_seconds=None if args.minutes is None else args.minutes * 60,
        pause_file=PAUSE_FILE,
    )


if __name__ == "__main__":
    main()
