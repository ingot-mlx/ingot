"""MLX reference of one layer's MoE block of Qwen3.6-35B-A3B (qwen3_5_moe) with real weights.

Realistic input: real text (the markdown files of --text-dir plus ingot's sources) tokenized and run through layers
0..L-1 of the model with mlx-lm; the MoE input of layer L is post_attention_layernorm(h + attn(h)).

Writes to <out>/:
  x.npy        MoE input in fp16 [rows, 2048] (it comes from bf16: the conversion is exact)
  y_f32.npy    "truth" output: fp32 router with the dequantized 8-bit weights, top-k over fp32,
               experts with gather_qmm in fp32 (exact 4-bit weights, fp32 activations)
  y_mlx.npy    mlx-lm's Qwen3NextSparseMoeBlock as is, in bf16 (as mlx-lm runs it)
  inds.npy     top-8 of the fp32 router [rows, 8] (int32, sorted by descending score)
  scores.npy   normalized top-8 weights (fp32)
  meta.json    shapes, versions and (with --bench) MLX timings per piece

Usage:
  .venv/bin/python moe_ref.py --layer 20 --rows 512 --bench
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.base import create_attention_mask

HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(model):
    model = os.path.expanduser(model)
    if os.path.isdir(model):
        return model
    base = os.path.join(HUB, "models--" + model.replace("/", "--"))
    rev = open(os.path.join(base, "refs", "main")).read().strip()
    return os.path.join(base, "snapshots", rev)


def corpus(text_dir):
    """Real, stable text: the markdown files of text_dir and ingot's sources, in a fixed order."""
    files = sorted(glob.glob(os.path.join(os.path.expanduser(text_dir), "*.md")))
    files += sorted(glob.glob(os.path.join(HERE, "..", "Sources", "**", "*.swift"), recursive=True))
    files += sorted(glob.glob(os.path.join(HERE, "..", "Sources", "Ingot", "Shaders", "*.metal")))
    return "\n\n".join(open(f, encoding="utf-8").read() for f in files)


def bench(f, args, warmup, iters):
    for _ in range(warmup):
        mx.eval(f(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(f(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return dict(median_ms=ts[len(ts) // 2], min_ms=ts[0], p90_ms=ts[int(len(ts) * 0.9)], n=len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Models/Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib")
    ap.add_argument("--text-dir", required=True, help="folder with the *.md files used as input text")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    d = resolve(a.model)
    out = a.out or os.path.join(HERE, "out", a.model.rstrip("/").split("/")[-1], f"moe_L{a.layer}_R{a.rows}")
    os.makedirs(out, exist_ok=True)
    model, tok = load(d, lazy=True)
    tm = model.language_model.model
    layers = tm.layers

    # Real input up to layer L (only the weights of those layers are materialized).
    ids = tok.encode(corpus(a.text_dir))
    assert len(ids) >= a.rows, f"corpus too short: {len(ids)} tokens"
    toks = mx.array(ids[: a.rows])[None]
    h = tm.embed_tokens(toks)
    fa_mask = create_attention_mask(h, None)
    t0 = time.time()
    for i in range(a.layer):
        ly = layers[i]
        h = ly(h, mask=None if ly.is_linear else fa_mask, cache=None)
        mx.eval(h)
    ly = layers[a.layer]
    r = ly.linear_attn(ly.input_layernorm(h), None, None) if ly.is_linear else ly.self_attn(ly.input_layernorm(h), fa_mask, None)
    xin = ly.post_attention_layernorm(h + r)[0]  # [rows, 2048] bf16
    mx.eval(xin)
    print(f"real input: {a.rows} tokens through {a.layer} layers in {time.time() - t0:.1f} s, dtype {xin.dtype}")

    blk = ly.mlp
    mx.eval(blk.parameters())
    E, K = blk.num_experts, blk.top_k
    gsz, bits = blk.gate.group_size, blk.gate.bits
    print(f"router {bits} bits g{gsz}, {E} experts, top-{K}, norm_topk_prob={blk.norm_topk_prob}")

    # 1) mlx-lm block as is (bf16)
    y_mlx = blk(xin)

    # 2) verdad en fp32
    x32 = xin.astype(mx.float32)

    def deq(m):
        return mx.dequantize(m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32),
                             group_size=m.group_size, bits=m.bits)

    logits = x32 @ deq(blk.gate).T
    probs = mx.softmax(logits, axis=-1)
    inds = mx.argsort(-probs, axis=-1)[:, :K]
    sc = mx.take_along_axis(probs, inds, axis=-1)
    if blk.norm_topk_prob:
        sc = sc / sc.sum(axis=-1, keepdims=True)
    sw = blk.switch_mlp

    def gq(xx, m, idx):
        return mx.gather_qmm(xx, m.weight, m.scales.astype(mx.float32), m.biases.astype(mx.float32),
                             rhs_indices=idx, transpose=True, group_size=m.group_size, bits=m.bits)

    xe = mx.expand_dims(x32, (-2, -3))  # [T,1,1,D]
    g = gq(xe, sw.gate_proj, inds)
    u = gq(xe, sw.up_proj, inds)
    hh = g * mx.sigmoid(g) * u
    ye = gq(hh, sw.down_proj, inds).squeeze(-2)  # [T,K,D]
    y_routed = (ye * sc[..., None]).sum(axis=-2)
    se = blk.shared_expert
    sg_ = x32 @ deq(se.gate_proj).T
    su_ = x32 @ deq(se.up_proj).T
    ys = (sg_ * mx.sigmoid(sg_) * su_) @ deq(se.down_proj).T
    sgate = mx.sigmoid(x32 @ deq(blk.shared_expert_gate).T)
    y_f32 = y_routed + sgate * ys
    mx.eval(y_mlx, y_f32, inds, sc)

    np.save(os.path.join(out, "x.npy"), np.array(xin.astype(mx.float16)))
    np.save(os.path.join(out, "y_f32.npy"), np.array(y_f32).astype(np.float32))
    np.save(os.path.join(out, "y_mlx.npy"), np.array(y_mlx.astype(mx.float32)).astype(np.float32))
    np.save(os.path.join(out, "inds.npy"), np.array(inds).astype(np.int32))
    np.save(os.path.join(out, "scores.npy"), np.array(sc).astype(np.float32))

    yr = np.array(y_f32)
    ym = np.array(y_mlx.astype(mx.float32))
    err = ym - yr
    psnr = 20 * np.log10(np.abs(yr).max() / np.sqrt((err ** 2).mean()))
    print(f"  y_mlx (bf16) vs y_f32: max|Δ| {np.abs(err).max():.3e}  L2 rel {np.linalg.norm(err) / np.linalg.norm(yr):.2e}  PSNR {psnr:.1f} dB")
    # Rows where mlx-lm's bf16 router picks a different top-8 set than the fp32 router.
    gb = mx.softmax(blk.gate(xin), axis=-1, precise=True)
    ib = np.sort(np.array(mx.argpartition(gb, kth=-K, axis=-1)[..., -K:]), axis=-1)
    flips = int((ib != np.sort(np.array(inds), axis=-1)).any(axis=-1).sum())
    print(f"  mlx-lm bf16 router: {flips} of {a.rows} rows with a different top-8 than fp32")
    cnt = np.bincount(np.array(inds).ravel(), minlength=E)
    print(f"  rows per expert: mean {cnt.mean():.1f}, median {np.median(cnt):.0f}, max {cnt.max()}, "
          f"expertos sin filas {(cnt == 0).sum()}, p90 {np.percentile(cnt, 90):.0f}")

    meta = dict(model=a.model, snapshot=d, layer=a.layer, rows=a.rows, hidden=int(xin.shape[1]), experts=E, top_k=K,
                moe_intermediate=int(sw.gate_proj.weight.shape[1]), mlx=mx.__version__, mlx_bf16_router_flips=flips,
                counts=dict(mean=float(cnt.mean()), median=float(np.median(cnt)), max=int(cnt.max()),
                            zero=int((cnt == 0).sum())))
    if a.bench:
        # Wall-clock times with mx.eval per iteration; weights already evaluated before compiling.
        def router(x):
            gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
            ii = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
            s = mx.take_along_axis(gates, ii, axis=-1)
            return ii, s / s.sum(axis=-1, keepdims=True)

        ii, ss = router(xin)
        mx.eval(ii, ss)

        def routed(x, ii, ss):
            return (blk.switch_mlp(x, ii) * ss[..., None]).sum(axis=-2)

        def shared(x):
            return mx.sigmoid(blk.shared_expert_gate(x)) * blk.shared_expert(x)

        # 10 independent blocks in a single graph: amortizes graph construction and launch
        # (it approaches MLX's GPU time per block).
        xs10 = [xin * (1 + 1e-3 * i) for i in range(10)]
        mx.eval(xs10)

        def block10(*xs):
            return [blk(x) for x in xs]

        times = {}
        r10 = bench(block10, xs10, a.warmup, max(a.iters // 5, 5))
        times["block_x10_per_block"] = {k: (v / 10 if k.endswith("_ms") else v) for k, v in r10.items()}
        print(f"  MLX block (10 per eval, per block): median {r10['median_ms'] / 10:.3f} ms")
        for name, f, args_ in (("block", blk, (xin,)), ("router", router, (xin,)),
                               ("routed", routed, (xin, ii, ss)), ("shared", shared, (xin,))):
            for comp in (False, True):
                ff = mx.compile(f) if comp else f
                key = name + ("_compiled" if comp else "")
                times[key] = bench(ff, args_, a.warmup, a.iters)
                print(f"  MLX {key}: median {times[key]['median_ms']:.3f} ms (min {times[key]['min_ms']:.3f})")
        meta["bench"] = times
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print("saved to", out)


if __name__ == "__main__":
    main()
