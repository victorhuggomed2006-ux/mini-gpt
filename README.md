# mini-gpt

A small **character-level GPT** (821,632 parameters) trained **from scratch, on a laptop CPU, in about 23 minutes**, on
public-domain Portuguese: eleven works by Machado de Assis (Project Gutenberg). It writes text in the style of
the author, and it **runs in the browser** with no server.

It does **not** understand what it writes: it continues a text, it does not answer questions. It also invents words.
Read the [model card](MODEL_CARD.md) for what it can and cannot do.

> *Capitu a um trabalha tratar, e me deixou menos á moça de seus finados o pequenos, não branco confiava o caso
> de um dos seus que os dias...* (first sample, fixed seed)

## Try it

Open `web/index.html` in a browser (a double-click is enough), type the start of a sentence and press Generate.
Nothing is uploaded: the weights (2.1 MB) are loaded from `artifacts/gpt_weights.js`.

## Reproduce it

```
uv sync                                          # PyTorch for CPU only
uv run python -m mini_gpt.train --steps 3000     # about 25 minutes, idle priority, 2 threads
uv run python -m mini_gpt.export                 # writes artifacts/gpt_weights.js and gpt_reference.json
node web_test/forward.test.mjs                   # JavaScript == Python reference
uv run python -m mini_gpt.generate "Capitu "     # continue a text
uv run pytest && uv run ruff check . && uv run mypy && uv run bandit -r src
```

The Machado de Assis texts are in `data/raw` (public domain). Without them, download from Project Gutenberg
and put the `.txt` files there.

## Gentle on your machine

Training is meant to run while you do something else (a call, a video):

- idle process priority (the operating system serves every other program first) and 2 threads;
- a checkpoint every 250 steps; `--resume` continues from it, `--minutes N` stops after N minutes;
- create a file named `PAUSE` in this folder to pause (for example, during a call), and delete it to continue.

## What is in the repository

| Path | What it is |
|---|---|
| `src/mini_gpt/` | data cleaning, model, training, export and generation (Python, fully typed) |
| `tests/` | 17 tests: the model, causal attention, the training loop, the export, the idle priority |
| `artifacts/` | trained weights (`gpt_model.pt`), browser weights (`gpt_weights.js`), reference outputs |
| `web/` | the browser demo and the hand-written JavaScript model |
| `web_test/` | proves the JavaScript reproduces the Python results |
| `data/raw/` | the Machado de Assis texts used for training |
| `MODEL_CARD.md` | what the model is, how it was trained, results, samples and limitations |
| `docs/ENGINEERING.md` | the decisions behind the project |

## License

- **Code:** GNU Affero General Public License v3.0 or later (`LICENSE`). If you modify and distribute it, or offer it as a
  service over a network, you must publish your changes under the same license.
- **Weights, model card and documentation:** Creative Commons Attribution-ShareAlike 4.0 (`LICENSE-WEIGHTS-AND-DOCS`).
- The training texts are in the public domain.

These licenses allow commercial use (no free-software license can forbid it); what they guarantee is that whoever
receives the work also receives the source and may share it freely.

Copyright (c) 2026 Victor Huggo de Medeiros.
