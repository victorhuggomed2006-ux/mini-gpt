# Model card: mini-gpt

A very small GPT that continues Portuguese text in the style of Machado de Assis. It is a teaching and
portfolio model, built to show the whole path of a language model: data, architecture, training,
evaluation and running on the device.

## Model

| | |
|---|---|
| Type | Decoder-only transformer (GPT), character level |
| Vocabulary | 93 characters (letters, digits, punctuation and accents found in the text) |
| Layers | 4 transformer blocks, 4 attention heads, embedding size 128, context of 128 characters |
| Parameters | 821,632 |
| Weights | `artifacts/gpt_model.pt` (PyTorch, 3.3 MB) and `artifacts/gpt_weights.js` (float16, 2.1 MB, for the browser) |
| Framework | PyTorch (CPU), with a hand-written JavaScript port |

## Intended use

- Learning and demonstration: see how a language model works end to end, train one on a laptop, run it in a page.
- Playing with text generation in the style of a 19th-century Brazilian author.

**Not intended for:** answering questions, giving advice, summarising, translating or any task that needs knowledge.
It has none: it only predicts the next character.

## Training data

Eleven works by Machado de Assis from [Project Gutenberg](https://www.gutenberg.org/) (public domain):
Dom Casmurro, Memorias Posthumas de Braz Cubas, Quincas Borba, Esau e Jacob, Memorial de Ayres,
Helena, Yaya Garcia, A Mao e a Luva, Papeis Avulsos, Historias Sem Data and Reliquias de Casa Velha.
3.66 million characters; the last 5% is the validation set. The texts use the **period spelling**
(for example "egreja", "ella", "cousa"), so the model writes in that spelling.

## Training

3,000 steps, batch of 32 windows of 128 characters (about 12 million characters seen, around 3.5 passes over
the data), AdamW (learning rate 3e-3, 100 steps of warm-up, cosine decay to 10%, weight decay 0.01), gradient clipping
at 1.0, seed 0. About 23 minutes on a 6-core CPU using 2 threads at idle priority.

## Results

| Measure | Value |
|---|---|
| Validation loss | 1.626 nats per character |
| In bits | 2.35 bits per character |
| Perplexity per character | 5.08 |
| JavaScript vs. Python (logits on a reference prompt) | difference below 0.00001 |
| JavaScript vs. Python (greedy continuation of 40 characters) | identical text |

For comparison, guessing uniformly among 93 characters would give a loss of 4.53.

## What it writes (samples, not cherry-picked: first runs with fixed seeds)

> **Capitu** a um trabalha tratar, e me deixou menos á moça de seus finados o pequenos, não branco confiava
> o caso de um dos seus que os dias, e se ella as acabas e as realitas todas na boa dos outras.

> **O dia amanheceu** que lhe lhe fizera um desejo gesto amigo, e eu lhe comprovaveis, e tenha o seu distingue.
> Tudo se viu.

It looks like Portuguese and keeps the pace, punctuation, names and rhythm of the author, but it also invents words
("realitas", "belicidade") and the sentences do not mean anything beyond a few words.

## Limitations (please read)

- **It does not understand.** It continues a text; it does not know facts and cannot answer questions.
- **Invented words and broken grammar**, because it works character by character with under a million parameters.
- **Period spelling and period views.** The training texts are from the 1880s to 1900s. They contain attitudes that are
  outdated or offensive today (about slavery, women, class and race), and the model can reproduce that kind of language.
- Short memory: it only sees the last 128 characters.
- No safety filtering of any kind.

## How to reproduce

```
uv sync                                          # PyTorch for CPU only
uv run python -m mini_gpt.train --steps 3000     # about 25 minutes, idle priority, 2 threads
uv run python -m mini_gpt.export                 # writes artifacts/gpt_weights.js and gpt_reference.json
node web_test/forward.test.mjs                   # JavaScript == Python reference
uv run python -m mini_gpt.generate "Capitu "     # continue a text
```

## License

Code: GNU AGPL-3.0-or-later (`LICENSE`). Weights, this card and the documentation: CC BY-SA 4.0
(`LICENSE-WEIGHTS-AND-DOCS`). The training texts are in the public domain. Copyright (c) 2026 Victor Huggo de Medeiros.
