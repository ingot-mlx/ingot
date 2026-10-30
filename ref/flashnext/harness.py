"""Reference harness of Qwen3.8-Flash-Next with the qwen4_exp modules of mlx-vlm, layer by layer, on our weights.

Implementation independent of the engine and of splash: instantiates mlx-vlm `Qwen4ExpDecoderLayer` one layer
at a time, loads our converted weights (scripts/flashnext) dequantized to fp32 into it and runs it over the whole
sequence; saves the streams after each layer and frees the layer. All in fp32 on the MLX CPU (needs neither the GPU nor
the model to fit):
  - dense part: mx.dequantize of our 4/6/8 bits to fp32; HC, indexer and norms in BF16 → fp32;
  - experts: from experts.bin, d·(q − 1) in fp32 (exact), only those used by the routing of the layer;
  - PLE: rows of ple.bin dequantized to fp32 (4 bits g32: q·s + b);
  - embedding and lm_head: dequantized by rows or by chunks of the vocabulary.

Usage (venv with mlx 0.32.4, numpy and tokenizers; the mlx-vlm checkout 9790231 via --mlxvlm):
  python -I harness.py --model <model-dir> --ids ids.npy --out DIR [--layers N] [--gen 64]
       [--save-streams all|none|0,1,47] [--text "…"] [--chunk 2048]
Outputs in DIR: ids.npy, streams_L{i}.npy (fp32 [T, 4·2560] after layer i), logits.npy (fp32 [T, vocab], all the
positions of the forced text, or those of --logits-rows), greedy.json (greedy tokens after the prompt) and meta.json.
With --layers N the model is truncated to N layers (the logits come from the final mix after layer N − 1).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--mlxvlm', default=os.path.expanduser('~/.cache/uv/git-v1/checkouts/7c6c527541f961e6/9790231'))
ap.add_argument('--ids', help='.npy or .json with the prompt ids')
ap.add_argument('--text', help='prompt text (model tokenizer.json, no template)')
ap.add_argument('--out', required=True)
ap.add_argument('--layers', type=int, default=0)
ap.add_argument('--gen', type=int, default=0)
ap.add_argument('--save-streams', default='all')
ap.add_argument('--logits-rows', default='all', help="'all', 'last', list of rows or @file.json with the list")
ap.add_argument('--kv8', action='store_true',
                help='simulates the engine int8 KV (symmetric, amax/127 scale per token and head, round half to even)')
ap.add_argument('--chunk', type=int, default=2048,
                help='rows per prefill chunk (0 = the whole prompt at once: fails from ~9.5K tokens)')
args = ap.parse_args()

R = os.path.join(args.mlxvlm, 'mlx_vlm')
# mlx-vlm packages without their __init__ (they pull in transformers, PIL and the server): only the model modules.
for name in ('', '.models', '.models.qwen4_exp', '.models.qwen3_5', '.models.qwen3_5_moe', '.models.qwen3_vl',
             '.speculative'):
    m = types.ModuleType('mlx_vlm' + name)
    m.__path__ = [R + name.replace('.', '/')]
    sys.modules['mlx_vlm' + name] = m

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

mx.set_default_device(mx.cpu)
import mlx_vlm.models.qwen4_exp.language as QL  # noqa: E402
from mlx_vlm.models.cache import ArraysCache  # noqa: E402
from mlx_vlm.models.qwen4_exp.config import TextConfig  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts', 'flashnext'))
import fnformat as F  # noqa: E402

MD = os.path.expanduser(args.model)
CFG = json.load(open(os.path.join(MD, 'config.json')))
FMT = CFG['ingot']
tc = dict(CFG['text_config'])
tcfg = TextConfig.from_dict(tc)
NL = args.layers or tcfg.num_hidden_layers
os.makedirs(args.out, exist_ok=True)

# ------------------------------------------------------------------ dense weights
INDEX = json.load(open(os.path.join(MD, 'model.safetensors.index.json')))['weight_map']
_files = {}


def tensor(name):
    fn = INDEX[name]
    if fn not in _files:
        _files[fn] = mx.load(os.path.join(MD, fn))
    return _files[fn][name]


def qparams(base):
    q = CFG['quantization']
    o = q.get(base, {})
    return o.get('group_size', q['group_size']), o.get('bits', q['bits'])


def dense_f32(base):
    """fp32 weight of a module (dequantized if it has scales)."""
    if base + '.scales' in INDEX:
        g, b = qparams(base)
        return mx.dequantize(tensor(base + '.weight'), tensor(base + '.scales').astype(mx.float32),
                             tensor(base + '.biases').astype(mx.float32), group_size=g, bits=b)
    return tensor(base + '.weight').astype(mx.float32)


def module_weights(prefix):
    """{relative path: fp32} of all the tensors under `prefix` (quantized → .weight fp32)."""
    out = {}
    for k in INDEX:
        if not k.startswith(prefix):
            continue
        rel = k[len(prefix):]
        if rel.endswith('.scales') or rel.endswith('.biases'):
            continue
        if rel.endswith('.weight'):
            out[rel] = dense_f32(k[:-len('.weight')])
        else:
            out[rel] = tensor(k).astype(mx.float32)
    return out


# ------------------------------------------------------------------ experts (experts.bin) and PLE (ple.bin)
EXP = np.memmap(os.path.join(MD, FMT['experts']['file']), dtype=np.uint8, mode='r')
PLE = np.memmap(os.path.join(MD, FMT['ple']['file']), dtype=np.uint8, mode='r')


def expert_f32(layer, e):
    """(gate, up, down) fp32 of an expert: d · (q − 1), exact."""
    rec = EXP[(layer * F.EXPERTS + e) * F.RECORD:(layer * F.EXPERTS + e + 1) * F.RECORD]
    p = F.record_planes(rec)
    return [mx.array(F.dequant_planar_q2(*p[n])) for n in ('gate', 'up', 'down')]


class LazyExperts(nn.Module):
    """Replaces SwitchGLU: only dequantizes the experts used by the routing (fp32, per expert)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.used = set()

    def __call__(self, x, inds):
        shp = x.shape
        xf = x.reshape(-1, shp[-1]).astype(mx.float32)
        ind = np.array(inds.reshape(-1, inds.shape[-1]))
        T, K = ind.shape
        out = np.zeros((T, K, shp[-1]), dtype=np.float32)
        for e in np.unique(ind):
            rows, slots = np.nonzero(ind == e)
            g, u, d = expert_f32(self.layer, int(e))
            xr = xf[mx.array(rows)]
            h = nn.silu(xr @ g.T) * (xr @ u.T)
            out[rows, slots] = np.array(h @ d.T)
            self.used.add(int(e))
        return mx.array(out).reshape(*shp[:-1], K, shp[-1])


def ple_rows_f32(idx):
    """Rows of the PLE table (fp32 [n, 160]) for global indices."""
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    rpp, rb, page = F.PLE_ROWS_PER_PAGE, F.PLE_ROW, F.PAGE
    offs = (idx // rpp) * page + (idx % rpp) * rb
    raw = PLE[offs[:, None] + np.arange(rb)[None]]  # [n, 100]
    w = raw[:, :80].copy().view('<u4')  # [n, 20]
    q = ((w[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(-1, 160).astype(np.float32)
    s = raw[:, 80:90].copy().view(np.uint16).astype(np.uint32) << 16
    b = raw[:, 90:100].copy().view(np.uint16).astype(np.uint32) << 16
    s = s.view(np.float32)
    b = b.view(np.float32)
    return q * np.repeat(s, 32, axis=1) + np.repeat(b, 32, axis=1)


class PLEReader(nn.Module):
    """Replaces ShardedEmbedding (128 BF16 chunks): reads the rows of ple.bin."""

    def __call__(self, indices):
        flat = np.array(indices).reshape(-1)
        rows = ple_rows_f32(flat)
        return mx.array(rows).reshape(*indices.shape, F.PLE_DIM)


_orig_sharded = QL.ShardedEmbedding.__init__


def _light_sharded(self, num_embeddings, dims, num_shards):
    nn.Module.__init__(self)  # without the 128 nn.Embedding (lazy, but 205 GB if anyone evaluates them)
    self.dims = dims


QL.ShardedEmbedding.__init__ = _light_sharded

if args.kv8:
    def _qd(x):
        x = x.astype(mx.float32)
        amax = mx.max(mx.abs(x), axis=-1, keepdims=True)
        sc = mx.where(amax > 0, amax / 127.0, mx.ones_like(amax))
        return mx.clip(mx.round(x / sc), -127, 127) * sc

    _base_uaf = QL.QSAKVCache.update_and_fetch
    QL.QSAKVCache.update_and_fetch = lambda self, k, v: _base_uaf(self, _qd(k), _qd(v))


# ------------------------------------------------------------------ layers
def build_layer(i):
    layer = QL.Qwen4ExpDecoderLayer(tcfg, i)
    pre = f'language_model.model.layers.{i}.'
    w = module_weights(pre)
    layer.mlp.switch_mlp = LazyExperts(i)
    if 'ple' in layer:
        layer.ple.ple_embedding.ngram_embedding = PLEReader()
        m = FMT['ple']
        emb = layer.ple.ple_embedding
        assert emb.layer_multipliers.tolist() == m['multipliers'], (emb.layer_multipliers.tolist(), m['multipliers'])
        assert emb.ngram_heads_offsets.tolist() == m['head_offsets']
        assert emb.ngram_heads_vocab_sizes.tolist() == m['head_vocab_sizes']
    have = dict(nn.utils.tree_flatten(layer.parameters()))
    # The PLE hash parameters are computed by mlx-vlm from the config (and checked above against ours).
    missing = [k for k in have if k not in w and 'switch_mlp' not in k and '.ple_embedding.' not in k]
    extra = [k for k in w if k not in have]
    if missing or extra:
        raise SystemExit(f'layer {i}: missing {missing[:8]} extra {extra[:8]}')
    for k, v in w.items():
        if tuple(have[k].shape) != tuple(v.shape):
            raise SystemExit(f'layer {i}: shape of {k} {v.shape} != {have[k].shape}')
    layer.load_weights(list(w.items()), strict=False)
    layer.eval()
    return layer


def make_cache(i):
    if tcfg.layer_types[i] == 'linear_attention':
        return ArraysCache(size=4 if (i + 1) in tcfg.ple_layer_ids else 2)
    return QL.QSAKVCache()


def embed(ids):
    base = 'language_model.model.embed_tokens'
    g, b = qparams(base)
    ix = mx.array(np.asarray(ids, dtype=np.int32))
    return mx.dequantize(tensor(base + '.weight')[ix], tensor(base + '.scales')[ix].astype(mx.float32),
                         tensor(base + '.biases')[ix].astype(mx.float32), group_size=g, bits=b)


MIXER = QL.Qwen4ExpGatedResidual(tcfg, use_combine=False)
MIXER.load_weights(list(module_weights('language_model.model.hyper_connection_mixer.').items()), strict=True)


def head_logits(h):
    """h fp32 [T, 4·2560] → logits fp32 [T, vocab] (final mix + lm_head dequantized by chunks)."""
    x = MIXER(h)
    base = 'language_model.lm_head'
    g, b = qparams(base)
    W, S, B = tensor(base + '.weight'), tensor(base + '.scales'), tensor(base + '.biases')
    outs = []
    for a in range(0, W.shape[0], 16384):
        wd = mx.dequantize(W[a:a + 16384], S[a:a + 16384].astype(mx.float32), B[a:a + 16384].astype(mx.float32),
                           group_size=g, bits=b)
        outs.append(x @ wd.T)
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=-1)


# ------------------------------------------------------------------ run
def load_ids():
    if args.ids:
        return [int(t) for t in (np.load(args.ids) if args.ids.endswith('.npy') else json.load(open(args.ids)))]
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(MD, 'tokenizer.json')).encode(args.text, add_special_tokens=False).ids


ids = load_ids()
np.save(os.path.join(args.out, 'ids.npy'), np.array(ids, dtype=np.int32))
save = set(range(NL)) if args.save_streams == 'all' else (
    set() if args.save_streams == 'none' else {int(x) for x in args.save_streams.split(',')})
caches = [make_cache(i) for i in range(NL)]
fa_idx = next(i for i in range(NL) if tcfg.layer_types[i] != 'linear_attention') if NL > 3 else None
ssm_idx = 0


def run(tokens, record):
    """Runs `tokens` (over the current caches) through the NL layers; returns the final streams [T, 10240].

    In chunks of --chunk rows, layer by layer (each layer is built once and runs all the chunks in order, over the
    cache left by the previous one): dense attention with an MLX mask on the CPU (the one the harness QSA uses) fails
    with the whole prompt between 9.5K and 10K tokens (bus error, or NaN and garbage in every row).
    """
    C = args.chunk if args.chunk > 0 else len(tokens)
    parts = [tokens[a:a + C] for a in range(0, len(tokens), C)]
    xs, inps, masks = [], [], []
    for p in parts:
        x = mx.tile(embed(p)[None], (1, 1, tcfg.hc_count))
        xs.append(x)
        inps.append(mx.array(np.asarray(p, dtype=np.int64))[None])
        # The masks ("causal" or None) depend only on the rows of the chunk, not on what is already in the cache.
        masks.append((QL._create_qwen4_exp_attention_mask(x, caches[fa_idx]) if fa_idx is not None else None,
                      QL._create_qwen3_5_ssm_mask(x, caches[ssm_idx])))
    used = {}
    for i in range(NL):
        t0 = time.time()
        layer = build_layer(i)
        lin = tcfg.layer_types[i] == 'linear_attention'
        for c in range(len(parts)):
            fa_mask, ssm_mask = masks[c]
            x = layer(xs[c], inps[c], mask=ssm_mask if lin else fa_mask, cache=caches[i], position_ids=None)
            mx.eval(x)
            for st in (caches[i].state if hasattr(caches[i], 'state') else []):
                if isinstance(st, mx.array):
                    mx.eval(st)
            xs[c] = x
        used[i] = len(layer.mlp.switch_mlp.used)
        if record and i in save:
            np.save(os.path.join(args.out, f'streams_L{i}.npy'), np.concatenate([np.array(x[0]) for x in xs], axis=0))
        if record:
            print(f'layer {i}: {time.time() - t0:.1f} s, {used[i]} experts', flush=True)
        del layer
    return mx.concatenate([x[0] for x in xs], axis=0), used


t0 = time.time()
h, used = run(ids, True)
if args.logits_rows.startswith('@'):
    rows = [int(x) for x in json.load(open(args.logits_rows[1:]))]
else:
    rows = list(range(len(ids))) if args.logits_rows == 'all' else (
        [len(ids) - 1] if args.logits_rows == 'last' else [int(x) for x in args.logits_rows.split(',')])
if rows[-1] != len(ids) - 1:
    rows.append(len(ids) - 1)  # the first greedy token always comes from the last row
logits = head_logits(h[mx.array(rows)])
np.save(os.path.join(args.out, 'logits.npy'), np.array(logits))
greedy = [int(mx.argmax(logits[-1]).item())]
print(f'prefill {len(ids)} tokens: {time.time() - t0:.0f} s; first token {greedy[0]}', flush=True)
for n in range(1, args.gen):
    hs, _ = run([greedy[-1]], False)
    lg = head_logits(hs)
    greedy.append(int(mx.argmax(lg[-1]).item()))
    print(f'token {n}: {greedy[-1]}', flush=True)
json.dump(greedy, open(os.path.join(args.out, 'greedy.json'), 'w'))
json.dump({'layers': NL, 'tokens': len(ids), 'logits_rows': rows, 'experts_used': used, 'gen': len(greedy),
           'mlxvlm': args.mlxvlm, 'kv8': args.kv8, 'seconds': time.time() - t0}, open(os.path.join(args.out, 'meta.json'), 'w'))
print('DONE', flush=True)
