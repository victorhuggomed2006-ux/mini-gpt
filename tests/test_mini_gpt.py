from __future__ import annotations

import math
from pathlib import Path

import torch

from mini_gpt.data import CharVocab, build_corpus, clean_book, split_corpus
from mini_gpt.model import GPT, GPTConfig
from mini_gpt.train import fit, get_batch, lr_at

RAW = (
    "Junk before\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
    "Primeira linha\nsegunda linha\n\nOutro paragrafo\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
)


def tiny_model(vocab_size: int = 12) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32))


def test_clean_book_removes_header_footer_and_reflows() -> None:
    assert clean_book(RAW) == "Primeira linha segunda linha\n\nOutro paragrafo"


def test_vocab_roundtrip_and_unknown_characters() -> None:
    vocab = CharVocab.from_text("ab c")
    assert vocab.decode(vocab.encode("abc ab")) == "abc ab"
    assert vocab.decode(vocab.encode("a?b")) == "a b"  # unknown character becomes a space


def test_build_corpus_drops_rare_characters(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text(
        "*** START OF X ***\n" + "a" * 100 + "Z\n*** END OF X ***", encoding="utf-8"
    )
    corpus = build_corpus(tmp_path)
    assert "Z" not in corpus and corpus.count("a") == 100


def test_split_keeps_order() -> None:
    vocab, train, val = split_corpus("abcd" * 50, val_fraction=0.1)
    assert len(train) + len(val) == 200 and vocab.size == 4


def test_initial_loss_is_close_to_uniform() -> None:
    model = tiny_model()
    x = torch.randint(0, 12, (4, 16))
    _, loss = model(x, x)
    assert loss is not None and abs(loss.item() - math.log(12)) < 0.5


def test_attention_is_causal() -> None:
    model = tiny_model().eval()
    x = torch.randint(0, 12, (1, 16))
    changed = x.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % 12  # change only the future
    a, _ = model(x)
    b, _ = model(changed)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)


def test_generate_extends_the_sequence() -> None:
    model = tiny_model()
    out = model.generate(torch.zeros((1, 3), dtype=torch.long), max_new_tokens=25)
    assert out.shape == (1, 28)  # longer than block_size: the context is cropped


def test_lr_schedule() -> None:
    assert lr_at(0, 1000, 1.0) < lr_at(99, 1000, 1.0) <= 1.0
    assert abs(lr_at(1000, 1000, 1.0) - 0.1) < 1e-6


def test_training_learns_a_repeating_pattern() -> None:
    pattern = torch.tensor([0, 1, 2, 3, 4] * 200, dtype=torch.long)
    model = tiny_model(5)
    generator = torch.Generator().manual_seed(0)
    x, y = get_batch(pattern, 8, 16, generator)
    _, before = model(x, y)
    final = fit(model, pattern, pattern, steps=120, batch_size=8, base_lr=5e-3, eval_every=60)
    assert before is not None and final < before.item() * 0.3
