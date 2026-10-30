"""Dense part of Flash-Next in our format: from the original BF16 (only the dense tensors, fetched by ranges).

Usage: python -I convert_dense.py <folder with the dense BF16 safetensors> <mlx-community config> <output> [--hc-bits 8|16] [--hc-group 64|32]
Rounding quantization (mx.quantize, identical to mlx_lm.convert without calibration), on the MLX CPU:
  - GDN projections, attention, shared expert and ple.key_proj: 4 bits g64;
  - router (mlp.gate), mlp.shared_expert_gate and ple.value_proj: 8 bits g64;  lm_head: 6 bits g64;  embedding: 4 bits g64;
  - hyper-connections (down, up and inject of all of them, also the final mix): 8 bits g64 by default;
    with --hc-bits 16, plain BF16 (the exactness reference). Their norms, fp32;
  - QSA indexer: plain BF16;
  - norms, conv, A_log and dt_bias: fp32 (exact from BF16).
Norms stay zero-centered as in the checkpoint (the engine and mlx-vlm add 1); conv becomes [C, 4, 1].
Writes model-0000i-of-0000n.safetensors, model.safetensors.index.json and config.json (with "metal_lab").
"""
import os
import sys
import json
import glob
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fnformat as F

mx.set_default_device(mx.cpu)
src, cfgp, out = sys.argv[1], sys.argv[2], sys.argv[3]
HC_BITS = int(sys.argv[sys.argv.index('--hc-bits') + 1]) if '--hc-bits' in sys.argv else 8
HC_GROUP = int(sys.argv[sys.argv.index('--hc-group') + 1]) if '--hc-group' in sys.argv else 64
assert HC_BITS in (8, 16) and HC_GROUP in (32, 64)
os.makedirs(out, exist_ok=True)

def rule(name):
    """(bits, group) or None to leave it in BF16."""
    if not name.endswith('.weight') or 'norm' in name or 'conv1d' in name:
        return None
    if 'hyper_connection' in name:
        return (8, HC_GROUP) if HC_BITS == 8 else None
    if 'indexer.index_qk_proj' in name:
        return None
    if name.endswith('mlp.gate.weight') or name.endswith('shared_expert_gate.weight') or 'ple.value_proj' in name:
        return (8, 64)
    if name.endswith('lm_head.weight'):
        return (6, 64)
    return (4, 64)

def rename(k):
    if k.startswith('model.language_model.'):
        return 'language_model.model.' + k[len('model.language_model.'):]
    if k == 'lm_head.weight':
        return 'language_model.lm_head.weight'
    raise ValueError(k)

ple_meta = {}
qcfg = {'group_size': 64, 'bits': 4, 'mode': 'affine'}
outw = {}
for p in sorted(glob.glob(os.path.join(src, '*.safetensors'))):
    for k0, v in mx.load(p).items():
        if '.ple.ple_embedding.' in k0:
            ple_meta[k0.rsplit('.', 1)[-1]] = np.array(v).astype(np.int64).tolist()
            continue
        k = rename(k0)
        if k.endswith('conv1d.weight'):
            assert v.ndim == 3 and v.shape[1] == 1, (k, v.shape)
            v = v.moveaxis(2, 1)
        r = rule(k)
        if r is None or v.ndim != 2:
            # Small weights (norms, conv, A_log, dt_bias) in fp32 (the engine adds the 1 of the norms in fp32, without
            # rounding to bf16); hyper-connections and indexer in BF16.
            small = 'norm' in k or 'conv1d' in k or k.endswith('A_log') or k.endswith('dt_bias')
            outw[k] = mx.array(np.array(v.astype(mx.float32)))
            if not small:
                outw[k] = outw[k].astype(v.dtype)
            mx.eval(outw[k])
            continue
        bits, gs = r
        wq, s, b = mx.quantize(v, group_size=gs, bits=bits)
        mx.eval(wq, s, b)
        base = k[:-len('.weight')]
        outw[base + '.weight'], outw[base + '.scales'], outw[base + '.biases'] = wq, s, b
        if (bits, gs) != (4, 64):
            qcfg[base] = {'group_size': gs, 'bits': bits, 'mode': 'affine'}
print(len(outw), 'output tensors;', 'PLE:', {k: v[:3] for k, v in ple_meta.items()})

# Shards of ~2 GB.
names = sorted(outw)
shards, cur, cursz = [], [], 0
for k in names:
    n = outw[k].nbytes
    if cur and cursz + n > 2_000_000_000:
        shards.append(cur); cur, cursz = [], 0
    cur.append(k); cursz += n
shards.append(cur)
index = {'metadata': {'total_size': sum(outw[k].nbytes for k in names)}, 'weight_map': {}}
for i, ks in enumerate(shards):
    fn = f'model-{i + 1:05d}-of-{len(shards):05d}.safetensors'
    mx.save_safetensors(os.path.join(out, fn), {k: outw[k] for k in ks}, metadata={'format': 'mlx'})
    for k in ks:
        index['weight_map'][k] = fn
json.dump(index, open(os.path.join(out, 'model.safetensors.index.json'), 'w'), indent=1)

cfg = json.load(open(cfgp))
cfg['quantization'] = qcfg
cfg['quantization_config'] = qcfg
tc = cfg['text_config']
cfg['metal_lab'] = {
    'format': 1,
    'experts': {'file': 'experts.bin', 'kind': 'q2_0_planar', 'record_bytes': F.RECORD, 'payload_bytes': F.PAYLOAD,
                'layers': F.LAYERS, 'experts': F.EXPERTS, 'hidden': F.HIDDEN, 'intermediate': F.INTER, 'group': F.GROUP,
                'offsets': {k: {'codes': c, 'scales': s} for k, (c, s) in F.OFFSETS.items()},
                'source': 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Q2_0 (lossless)'},
    'ple': {'file': 'ple.bin', 'rows': 320001536, 'dim': F.PLE_DIM, 'bits': 4, 'group': F.PLE_GROUP,
            'row_bytes': F.PLE_ROW, 'rows_per_page': F.PLE_ROWS_PER_PAGE, 'page_bytes': F.PAGE,
            'layer': tc['ple_layer_ids'][0] - 1,
            'multipliers': ple_meta['layer_multipliers'], 'head_offsets': ple_meta['ngram_heads_offsets'],
            'head_vocab_sizes': ple_meta['ngram_heads_vocab_sizes'], 'eos_token_id': tc['eos_token_id'],
            'source': 'mlx-community/Qwen3.8-Flash-Next-4bit (4 bits g32, not requantized)'},
    'dense': {'source': 'Qwen/Qwen3.8-Flash-Next BF16', 'method': 'rounding (mx.quantize), uncalibrated',
              'hc_bits': HC_BITS, 'hc_group': HC_GROUP},
}
json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
print('DONE', len(shards), 'shards', index['metadata']['total_size'] / 1e9, 'GB')
