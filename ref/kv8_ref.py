"""Reference with the lab's int8 KV cache emulated in mlx-lm: every K and V row that enters the cache is
quantized and dequantized with the engine's scheme (vattn_prework_i8 in Shaders/verify_attn_nax.metal): symmetric,
one fp32 scale per token and KV head, sc = amax / 127 (1 when amax = 0), q = clamp(rint(x / sc), -127, 127),
x_hat = q * sc. K is taken after the QK norm and the RoPE (what mlx-lm hands to the cache) and quantized in the
basis of the orthonormal Hadamard H (the engine stores H k and rotates q too: variant hadk, the default), V as it is. The cache
returns the keys of the current chunk too, so the prefill also reads the quantized K/V, as pattn_i8 does.

The cache classes are patched (KVCache and RotatingKVCache, the ones of the full and sliding attention layers); the
linear-attention state (ArraysCache) is left alone. Then the wrapped script runs unchanged: gen_ref.py, logits_ref.py
or another wrapper (yarn_ref.py), which can be chained.

GLM's MLA (glm4_moe_lite) keeps a latent cache instead: its attention is patched to the engine's layout
(emulate_mla: one 576 head [k_pe | c] in the H64 ⊕ H512 basis with two scales, one for k_pe and one for c, and V read
from K: V = the dequantized c of K, no copy of its own); the variant does not apply to it, except its own items (to
locate the loss or reproduce the former layout): mla1s (one scale for the 576 of K), mlavsep (V as its own int8 copy
of c with its scale), mlak16 (K, and so V, in fp16). The engine's former layout is mla1s+mlavsep.

--kv8-variant SPEC (optional, removed from the arguments): other KV schemes, to measure them before writing
kernels. SPEC is a '+'-separated list of:
  head    one symmetric scale per token and KV head, K unrotated (the engine's former scheme, before the Hadamard rotation)
  gN      one fp32 scale per group of N contiguous values of the head (N divides head_dim), e.g. g64, g128
  asym    asymmetric (affine): sc = (max - min) / 255 and min in fp32, q = rint((x - min) / sc) in [0, 255],
          x_hat = q * sc + min (sc = 1 when max = min); asym+g64 is mlx-lm's 8-bit QuantizedKVCache
  hadk    K_hat = H kv8(H k), H the orthonormal Walsh-Hadamard matrix of head_dim: the same as rotating Q and K
          after the RoPE and storing the rotated K, since (H q) . (H k) = q . k. The engine's scheme (default)
  hadv    the same with V (the engine would rotate the attention output back)
  win16   the sliding-window layers (RotatingKVCache) keep K and V in fp16; the full layers use the rest of SPEC
  fp16    every layer keeps K and V in fp16: the ceiling of what a better 8-bit scheme can reach, not a candidate
--kv8-group N is kept as an alias of --kv8-variant gN (unrotated).

Usage:
  .venv/bin/python kv8_ref.py logits_ref.py --f32 --model ~/.cache/metal-lab/toy/toy-granite --pdir prompts-toy ...
  .venv/bin/python kv8_ref.py --kv8-variant hadk+g64 yarn_ref.py logits_ref.py --f32 --model .../toy-gptoss ...
  .venv/bin/python kv8_ref.py --selftest
In-process (a driver that loads the model once): kv8_ref.install() (default "hadk"), then kv8_ref.set_variant(...) between runs;
set_variant(None) turns the emulation off (plain mlx-lm).
"""
import atexit
import os
import runpy
import sys

import mlx.core as mx
from mlx_lm.models import cache as kvcache

HERE = os.path.dirname(os.path.abspath(__file__))
FLAGS = ("head", "asym", "hadk", "hadv", "win16", "fp16", "mla1s", "mlavsep", "mlak16")


def parse_variant(spec: str) -> dict:
    """'hadk+g64' -> {'group': 64, 'asym': False, 'hadk': True, ...}; raises ValueError on an unknown item."""
    v = {"group": 0, **{f: False for f in FLAGS}, "spec": spec}
    for item in spec.split("+"):
        if item in FLAGS:
            v[item] = True
        elif item.startswith("g") and item[1:].isdigit() and int(item[1:]) > 0:
            v["group"] = int(item[1:])
        else:
            raise ValueError(f"unknown KV variant item '{item}' in '{spec}'")
    return v


def kv8(x: mx.array, group: int = 0, asym: bool = False) -> mx.array:
    """Quantize and dequantize x [..., head_dim] in fp32 like the engine's int8 KV (per group of `group` values if
    > 0; affine with 256 levels if `asym`). Returns fp32."""
    shape = x.shape
    xf = x.astype(mx.float32)
    if group:
        assert shape[-1] % group == 0, f"group {group} does not divide head_dim {shape[-1]}"
        xf = xf.reshape(*shape[:-1], shape[-1] // group, group)
    one = mx.array(1.0, mx.float32)
    if asym:
        lo = mx.min(xf, axis=-1, keepdims=True)
        span = mx.max(xf, axis=-1, keepdims=True) - lo
        sc = mx.where(span > 0, span / 255.0, one)
        q = mx.clip(mx.round((xf - lo) / sc), 0, 255)  # mx.round is round-half-to-even, like Metal's rint
        y = q * sc + lo
    else:
        amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
        sc = mx.where(amax > 0, amax / 127.0, one)
        y = mx.clip(mx.round(xf / sc), -127, 127) * sc
    return y.reshape(shape)


def had(x: mx.array) -> mx.array:
    """Orthonormal Walsh-Hadamard transform of the last axis in fp32 (H is symmetric and H H = I)."""
    return mx.hadamard_transform(x.astype(mx.float32))


def f16(x: mx.array) -> mx.array:
    return x.astype(mx.float16).astype(x.dtype)


def emulate(keys: mx.array, values: mx.array, v: dict, window: bool) -> tuple:
    """K and V as the cache would give them back with variant v (window: the layer is a sliding-window one)."""
    if v["fp16"] or (window and v["win16"]):
        return f16(keys), f16(values)

    def one(x, rotate):
        y = had(kv8(had(x), v["group"], v["asym"])) if rotate else kv8(x, v["group"], v["asym"])
        return y.astype(x.dtype)

    return one(keys, v["hadk"]), one(values, v["hadv"])


def emulate_mla(c: mx.array, k_pe: mx.array, v: dict = None) -> tuple:
    """GLM's latent cache as the engine stores it (Layers/MLA.swift): one KV head of 576 per token.
    K = [k_pe | c] (k_pe after the RoPE, in the engine's de-interleaved order) in the basis of the block Hadamard
    H64 ⊕ H512 (at_fwht), one int8 scale for k_pe and one for c; V is K's c (the engine reads it from kc and rotates
    the output back). Returns (c of K, k_pe of K, c of V), fp32, in mlx-lm's order."""
    n = k_pe.shape[-1]
    de = mx.array([2 * i for i in range(n // 2)] + [2 * i + 1 for i in range(n // 2)])
    inv = mx.argsort(de)
    v = v or {}
    k = mx.concatenate([had(k_pe[..., de]), had(c)], axis=-1)
    if v.get("mlak16"):
        k = f16(k)
    elif v.get("mla1s"):
        k = kv8(k)
    else:
        k = mx.concatenate([kv8(k[..., :n]), kv8(k[..., n:])], axis=-1)
    kpe = had(k[..., :n])[..., inv]
    ck = had(k[..., n:])
    cv = kv8(c.astype(mx.float32)) if v.get("mlavsep") else ck
    return ck, kpe, cv


def _patch_mla() -> None:
    """mlx-lm's Glm4MoeLiteAttention with the engine's latent int8 cache (emulate_mla): the cache stores
    [c of K | c of V] as its keys and k_pe of K as its values; the scores and P·V read each one (decode absorbed,
    prefill decompressed from the same latents, as mlx-lm does)."""
    try:
        from mlx_lm.models import glm4_moe_lite as g
    except ImportError:
        return
    from mlx_lm.models.base import scaled_dot_product_attention

    def call(self, x, mask=None, cache=None):
        if STATE["variant"] is None:
            return orig(self, x, mask, cache)
        B, L, _ = x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        ckv = self.kv_a_proj_with_mqa(x)
        ckv, k_pe = mx.split(ckv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
        lat = self.kv_a_layernorm(ckv)
        offset = cache.offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        lat = mx.expand_dims(lat, axis=1)
        ck, kpe, cv = emulate_mla(lat, k_pe, STATE["variant"])
        STATE["calls"] += 1
        dt = lat.dtype
        both, kpe = mx.concatenate([ck, cv], axis=-1).astype(dt), kpe.astype(dt)
        if cache is not None:
            cache.mla_latent = True  # already emulated: the patched update_and_fetch stores it as it is
            both, kpe = cache.update_and_fetch(both, kpe)
        ck, cv = mx.split(both, [self.kv_lora_rank], axis=-1)
        pe_scores = (q_pe * self.scale) @ kpe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype))
        if L == 1:
            q_nope = self.embed_q(q_nope)
            k, v = ck, cv
        else:
            k = self.embed_q(ck, transpose=False)
            v = self.unembed_out(cv)
        out = scaled_dot_product_attention(q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores)
        if L == 1:
            out = self.unembed_out(out)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    orig = g.Glm4MoeLiteAttention.__call__
    g.Glm4MoeLiteAttention.__call__ = call


STATE = {"variant": None, "calls": 0}


def set_variant(spec) -> None:
    STATE["variant"] = parse_variant(spec) if spec else None


def _patch(cls, window: bool) -> None:
    orig = cls.update_and_fetch

    def update_and_fetch(self, keys, values):
        v = STATE["variant"]
        if v is not None and not getattr(self, "mla_latent", False):
            STATE["calls"] += 1
            keys, values = emulate(keys, values, v, window)
        return orig(self, keys, values)

    cls.update_and_fetch = update_and_fetch


_INSTALLED = [False]


def install(spec="hadk") -> None:
    """Patch the cache classes (once) and select the variant."""
    if not _INSTALLED[0]:
        _patch(kvcache.KVCache, window=False)
        _patch(kvcache.RotatingKVCache, window=True)
        _patch_mla()
        _INSTALLED[0] = True
    set_variant(spec)


def _take(flag: str):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    val = sys.argv[i + 1]
    del sys.argv[i: i + 2]
    return val


def _check_scheme(x: mx.array, y: mx.array, group: int, asym: bool, tag: str) -> float:
    """y = kv8(x): every element within half a step of x, levels in range. Returns max |err| / step."""
    hd = x.shape[-1]
    g = group or hd
    xs, ys = x.reshape(*x.shape[:-1], hd // g, g), y.reshape(*x.shape[:-1], hd // g, g)
    if asym:
        lo = mx.min(xs, axis=-1, keepdims=True)
        span = mx.max(xs, axis=-1, keepdims=True) - lo
        sc = mx.where(span > 0, span / 255.0, 1.0)
        levels = mx.round((ys - lo) / sc)
        assert float(mx.min(levels).item()) >= 0 and float(mx.max(levels).item()) <= 255, f"{tag}: level out of [0, 255]"
    else:
        amax = mx.max(mx.abs(xs), axis=-1, keepdims=True)
        sc = mx.where(amax > 0, amax / 127.0, 1.0)
        assert float(mx.max(mx.abs(mx.round(ys / sc))).item()) <= 127, f"{tag}: level out of [-127, 127]"
    worst = float(mx.max(mx.abs(ys - xs) / sc).item())
    assert worst <= 0.5 + 1e-3, f"{tag}: error {worst:.5f} steps > 0.5"
    return worst


def selftest() -> None:
    mx.random.seed(0)
    specs = ["head", "g64", "g128", "g32", "asym", "asym+g64", "hadk", "hadv", "hadk+hadv", "hadk+g64", "hadk+asym",
             "win16", "hadk+win16", "fp16"]
    for hd in (64, 128, 256):
        k = mx.random.normal((2, 4, 33, hd)) * mx.random.uniform(0.01, 10.0, (2, 4, 33, 1))
        k[..., 3] = k[..., 3] * 40.0  # an outlier channel, like the real K
        k[0, 0, 0] = 0.0
        vv = mx.random.normal((2, 4, 33, hd)) + 0.5
        vv[0, 0, 0] = 0.0
        q = mx.random.normal((2, 4, 33, hd))
        for spec in specs:
            v = parse_variant(spec)
            if v["group"] and hd % v["group"]:
                continue
            worst = 0.0
            for window in (False, True):
                ek, ev = emulate(k, vv, v, window)
                tag = f"hd {hd} {spec}{' window' if window else ''}"
                assert float(mx.max(mx.abs(ek[0, 0, 0])).item()) == 0.0, f"{tag}: an all-zero K row must stay zero"
                assert float(mx.max(mx.abs(ev[0, 0, 0])).item()) == 0.0, f"{tag}: an all-zero V row must stay zero"
                if v["fp16"] or (window and v["win16"]):
                    assert mx.array_equal(ek, f16(k)).item() and mx.array_equal(ev, f16(vv)).item(), f"{tag}: not fp16"
                    continue
                for x, e, rot in ((k, ek, v["hadk"]), (vv, ev, v["hadv"])):
                    # With a rotation the bound holds in the rotated basis: H x_hat = kv8(H x) (up to fp32 rounding).
                    xr, er = (had(x), had(e)) if rot else (x, e)
                    worst = max(worst, _check_scheme(xr, er, v["group"], v["asym"], tag))
            if v["hadk"]:
                s0 = mx.sum(q * k, axis=-1)
                s1 = mx.sum(had(q) * had(k), axis=-1)
                rel = float((mx.max(mx.abs(s1 - s0)) / mx.max(mx.abs(s0))).item())
                assert rel < 1e-5, f"hd {hd} {spec}: (Hq).(Hk) differs from q.k by {rel:.2e}"
            print(f"selftest hd {hd:3d} {spec:11s}: max |err| / step = {worst:.4f} (<= 0.5), ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        selftest()
        sys.exit(0)
    group, spec = _take("--kv8-group"), _take("--kv8-variant")
    if group and spec:
        sys.exit("kv8_ref.py: --kv8-group and --kv8-variant together; use --kv8-variant gN+...")
    spec = spec or (f"g{group}" if group else "hadk")
    install(spec)
    atexit.register(lambda: print(f"kv8_ref.py: {STATE['calls']} cache updates emulated (variant {spec})",
                                  file=sys.stderr))
    script = os.path.join(HERE, sys.argv[1])
    sys.argv = [script] + sys.argv[2:]
    runpy.run_path(script, run_name="__main__")
