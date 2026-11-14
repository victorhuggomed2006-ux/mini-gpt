# How it was built (engineering notes)

## 1. The model

A GPT in miniature, written from scratch in PyTorch (`src/mini_gpt/model.py`, about 120 lines):

- **Character level.** The vocabulary is the 93 characters of the text, so there is no tokenizer to
  train and nothing to download. It also keeps the model tiny.
- **4 transformer blocks**, 4 attention heads, embedding size 128, context of 128 characters,
  **821,632 parameters**. Pre-norm blocks, GELU, learned position embeddings, and the output layer shares its
  weights with the input embedding (weight tying).
- **Causal attention** through PyTorch's `scaled_dot_product_attention(is_causal=True)`. A test proves that
  changing a future character never changes the predictions for earlier positions.

## 2. The data

Eleven novels and story collections by Machado de Assis from Project Gutenberg (public domain):
3.66 million characters. `data.py` removes the Gutenberg header and footer, re-flows hard-wrapped lines into
paragraphs and drops characters that appear fewer than 30 times. The last 5% of the text is the
validation set. The texts keep the **period spelling** (for example "egreja" and "attitude"), so the model
writes in that spelling.

## 3. Training on a machine you are also using

The goal was a model you can train on a laptop **while doing something else**:

- idle process priority (the operating system gives the CPU to every other program first);
- 2 threads only;
- a checkpoint every 250 steps and `--resume`, so it can be stopped any time;
- a `PAUSE` file: while it exists, training waits.

On a 6-core CPU it takes about half a second per step, so 3,000 steps take about half an hour.
Optimiser: AdamW, learning rate 3e-3 with 100 steps of warm-up and cosine decay to 10%, gradient clipping at 1.0.

## 4. The browser version

`web/gpt.js` runs the same model with no libraries. Weights are float16 (half the size of float32) in a JavaScript
file. A key/value cache makes each new character cost one forward pass.
`web_test/forward.test.mjs` runs the JavaScript on a reference prompt and checks that the logits match the
Python ones and that the greedy continuation is **the same text**.

One deliberate simplification: when the 128-character context is full, the browser keeps the most recent
half and rebuilds the cache, instead of re-reading the last 128 characters at every step as the Python
version does. This makes generation much faster; the two agree for the first 128 characters.

## 5. Quality gates

`ruff`, `mypy --strict`, `bandit`, `pytest` with coverage, and in CI also `pip-audit`, `gitleaks` and CodeQL.
Training is seeded.

## 6. What it cannot do

It continues text in the style it learned. It has no knowledge, no instructions and no safeguards,
and it cannot answer questions. Sentences look like Portuguese but, over more than a few lines, they do
not mean anything. See the [model card](../MODEL_CARD.md).
