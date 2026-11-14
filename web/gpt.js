/* Forward pass and sampling of the mini-GPT, written by hand (no libraries).
 * Weights come from mini-gpt (PyTorch), exported as float16 + base64 in window.GPT_WEIGHTS.
 * A key/value cache makes each new character cost one pass instead of a full recomputation.
 * When the context is full it keeps the most recent half and rebuilds the cache (a documented
 * simplification of the Python version, which re-reads the last block_size characters every step). */
(function (root) {
  'use strict';

  function halfToFloat(h) {
    var s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  function decode(b64) {
    var bin = atob(b64), n = bin.length >> 1, out = new Float32Array(n);
    for (var i = 0; i < n; i++) out[i] = halfToFloat(bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8));
    return out;
  }
  /* erf with ~1e-7 error (Abramowitz and Stegun 7.1.26 refined), enough for GELU */
  function erf(x) {
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function gelu(x) { return 0.5 * x * (1 + erf(x * 0.7071067811865476)); }

  function layerNorm(x, w, b, out) {
    var n = x.length, mean = 0, v = 0, i;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (i = 0; i < n; i++) { var d = x[i] - mean; v += d * d; }
    var inv = 1 / Math.sqrt(v / n + 1e-5);
    for (i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * w[i] + b[i];
  }
  /* y = W x + b, with W stored row-major (out, in) */
  function linear(x, W, b, outN, out) {
    var inN = x.length;
    for (var o = 0; o < outN; o++) {
      var sum = b ? b[o] : 0, base = o * inN;
      for (var i = 0; i < inN; i++) sum += W[base + i] * x[i];
      out[o] = sum;
    }
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function load(payload) {
    var cfg = payload.config, E = cfg.n_embd, H = cfg.n_head, D = E / H, L = cfg.n_layer, B = cfg.block_size, V = cfg.vocab_size;
    var T = {};
    Object.keys(payload.tensors).forEach(function (k) { T[k] = decode(payload.tensors[k].data); });
    var chars = payload.chars, index = {};
    for (var i = 0; i < chars.length; i++) index[chars[i]] = i;
    var space = index[' '] || 0;

    function encode(text) { var ids = []; for (var i = 0; i < text.length; i++) { var c = index[text[i]]; ids.push(c === undefined ? space : c); } return ids; }
    function decodeIds(ids) { var s = ''; for (var i = 0; i < ids.length; i++) s += chars[ids[i]]; return s; }

    function newCache() {
      var layers = [];
      for (var l = 0; l < L; l++) layers.push({ k: new Float32Array(B * E), v: new Float32Array(B * E) });
      return { layers: layers, len: 0 };
    }
    var x = new Float32Array(E), h = new Float32Array(E), qkv = new Float32Array(3 * E), att = new Float32Array(E),
        proj = new Float32Array(E), mlp = new Float32Array(4 * E), mlp2 = new Float32Array(E), scores = new Float32Array(B), logits = new Float32Array(V);

    /* one token at position cache.len; returns the logits (shared buffer) */
    function step(cache, token) {
      var pos = cache.len, i, l;
      for (i = 0; i < E; i++) x[i] = T['tok_emb.weight'][token * E + i] + T['pos_emb.weight'][pos * E + i];
      for (l = 0; l < L; l++) {
        var p = 'blocks.' + l + '.', C = cache.layers[l];
        layerNorm(x, T[p + 'ln1.weight'], T[p + 'ln1.bias'], h);
        linear(h, T[p + 'attn.qkv.weight'], T[p + 'attn.qkv.bias'], 3 * E, qkv);
        for (i = 0; i < E; i++) { C.k[pos * E + i] = qkv[E + i]; C.v[pos * E + i] = qkv[2 * E + i]; }
        for (var hd = 0; hd < H; hd++) {
          var off = hd * D, max = -Infinity, t, d, sum = 0;
          for (t = 0; t <= pos; t++) {
            var dot = 0;
            for (d = 0; d < D; d++) dot += qkv[off + d] * C.k[t * E + off + d];
            scores[t] = dot / Math.sqrt(D);
            if (scores[t] > max) max = scores[t];
          }
          for (t = 0; t <= pos; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
          for (d = 0; d < D; d++) {
            var acc = 0;
            for (t = 0; t <= pos; t++) acc += scores[t] * C.v[t * E + off + d];
            att[off + d] = acc / sum;
          }
        }
        linear(att, T[p + 'attn.proj.weight'], T[p + 'attn.proj.bias'], E, proj);
        for (i = 0; i < E; i++) x[i] += proj[i];
        layerNorm(x, T[p + 'ln2.weight'], T[p + 'ln2.bias'], h);
        linear(h, T[p + 'mlp.0.weight'], T[p + 'mlp.0.bias'], 4 * E, mlp);
        for (i = 0; i < 4 * E; i++) mlp[i] = gelu(mlp[i]);
        linear(mlp, T[p + 'mlp.2.weight'], T[p + 'mlp.2.bias'], E, mlp2);
        for (i = 0; i < E; i++) x[i] += mlp2[i];
      }
      layerNorm(x, T['ln_f.weight'], T['ln_f.bias'], h);
      linear(h, T['tok_emb.weight'], null, V, logits);
      cache.len = pos + 1;
      return logits;
    }

    /* feed ids into a fresh cache and return the logits after the last one */
    function prefill(cache, ids) {
      var out = null;
      for (var i = 0; i < ids.length; i++) out = step(cache, ids[i]);
      return out;
    }

    function sample(lg, temperature, topK, rand) {
      var n = lg.length, idx = [], i;
      for (i = 0; i < n; i++) idx.push(i);
      if (topK && topK < n) { idx.sort(function (a, b) { return lg[b] - lg[a]; }); idx = idx.slice(0, topK); }
      var max = -Infinity;
      for (i = 0; i < idx.length; i++) if (lg[idx[i]] > max) max = lg[idx[i]];
      var t = Math.max(temperature, 1e-6), probs = [], sum = 0;
      for (i = 0; i < idx.length; i++) { var p = Math.exp((lg[idx[i]] - max) / t); probs.push(p); sum += p; }
      var r = rand() * sum, acc = 0;
      for (i = 0; i < idx.length; i++) { acc += probs[i]; if (r <= acc) return idx[i]; }
      return idx[idx.length - 1];
    }

    /* one text session: next() returns the next generated character.
     * opts: temperature, topK, seed */
    function session(prompt, opts) {
      opts = opts || {};
      var ids = encode(prompt); if (!ids.length) ids = [space];
      var rand = mulberry32(opts.seed === undefined ? (Math.random() * 4294967296) >>> 0 : opts.seed);
      var cache = newCache(), lg = prefill(cache, ids.slice(-(B - 1))), out = ids.slice();
      var temperature = opts.temperature === undefined ? 0.8 : opts.temperature, topK = opts.topK === undefined ? 40 : opts.topK;
      return {
        next: function () {
          var tok = sample(lg, temperature, topK, rand);
          out.push(tok);
          if (cache.len >= B) { cache = newCache(); lg = prefill(cache, out.slice(-(B >> 1))); }
          else lg = step(cache, tok);
          return chars[tok];
        },
        text: function () { return decodeIds(out); }
      };
    }

    /* convenience: generate opts.length characters at once; calls onChar(char) for each */
    function generate(prompt, opts, onChar) {
      opts = opts || {};
      var s = session(prompt, opts);
      for (var n = 0; n < (opts.length || 300); n++) {
        if (opts.shouldStop && opts.shouldStop()) break;
        var c = s.next();
        if (onChar) onChar(c);
      }
      return s.text();
    }

    return { encode: encode, decode: decodeIds, newCache: newCache, step: step, prefill: prefill, session: session, generate: generate, config: cfg, valLoss: payload.val_loss, chars: chars };
  }

  var api = { load: load, decode: decode };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.GPTNet = api;
})(typeof window !== 'undefined' ? window : globalThis);
