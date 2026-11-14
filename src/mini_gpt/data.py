"""Corpus preparation and a character-level vocabulary."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

START_MARK = "*** START OF"
END_MARK = "*** END OF"
MIN_CHAR_COUNT = 30  # characters rarer than this are dropped to keep the vocabulary small
UNKNOWN = " "


def clean_book(raw: str) -> str:
    """Remove the Project Gutenberg header/footer and re-flow hard-wrapped lines into paragraphs."""
    text = raw.replace("\r\n", "\n").replace("﻿", "")
    start = text.find(START_MARK)
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    end = text.find(END_MARK)
    if end != -1:
        text = text[:end]
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        line = re.sub(r"\s+", " ", block).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def build_corpus(raw_dir: Path) -> str:
    """Clean every ``*.txt`` in ``raw_dir`` and join them, dropping very rare characters."""
    books = [clean_book(p.read_text(encoding="utf-8")) for p in sorted(raw_dir.glob("*.txt"))]
    corpus = "\n\n".join(books)
    counts = Counter(corpus)
    rare = {ch for ch, n in counts.items() if n < MIN_CHAR_COUNT}
    return "".join(UNKNOWN if ch in rare else ch for ch in corpus)


@dataclass(frozen=True)
class CharVocab:
    """Maps characters to integer ids and back."""

    chars: str

    @classmethod
    def from_text(cls, text: str) -> CharVocab:
        """Build the vocabulary from the characters present in ``text``."""
        return cls("".join(sorted(set(text))))

    @property
    def size(self) -> int:
        """Number of distinct characters."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Turn text into ids. Unknown characters become a space."""
        lookup = {ch: i for i, ch in enumerate(self.chars)}
        fallback = lookup.get(UNKNOWN, 0)
        return [lookup.get(ch, fallback) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Turn ids back into text."""
        return "".join(self.chars[i] for i in ids)


def split_corpus(
    corpus: str, val_fraction: float = 0.05
) -> tuple[CharVocab, torch.Tensor, torch.Tensor]:
    """Return the vocabulary and the train/validation ids (the last part is validation)."""
    vocab = CharVocab.from_text(corpus)
    data = torch.tensor(vocab.encode(corpus), dtype=torch.long)
    cut = int(len(data) * (1 - val_fraction))
    return vocab, data[:cut], data[cut:]
