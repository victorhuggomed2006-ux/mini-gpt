// Checks that the hand-written JavaScript model reproduces the Python reference.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const GPTNet = require(path.join(here, '..', 'web', 'gpt.js'));

const sandbox = {};
new Function('window', readFileSync(path.join(here, '..', 'artifacts', 'gpt_weights.js'), 'utf8'))(sandbox);
const net = GPTNet.load(sandbox.GPT_WEIGHTS);
const ref = JSON.parse(readFileSync(path.join(here, '..', 'artifacts', 'gpt_reference.json'), 'utf8'));

// 1) logits after the prompt
const cache = net.newCache();
const logits = net.prefill(cache, ref.prompt_ids);
const worst = Math.max(...ref.last_logits.map((v, i) => Math.abs(v - logits[i])));

// 2) greedy continuation must equal the Python one (same number of characters)
const text = net.generate(ref.prompt, { temperature: 1e-6, topK: 1, length: ref.greedy_ids.length - ref.prompt_ids.length, seed: 0 });
const sameText = text === ref.greedy_text;

console.log(`worst logit difference=${worst.toExponential(2)}  greedy text identical=${sameText}`);
if (!sameText) console.log('JS    :', JSON.stringify(text), '\nPython:', JSON.stringify(ref.greedy_text));
if (worst > 5e-3 || !sameText) { console.error('FAIL'); process.exit(1); }
console.log('OK: JavaScript matches the Python reference');
