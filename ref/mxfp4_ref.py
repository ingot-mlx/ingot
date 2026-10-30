#!/usr/bin/env python3
"""Reference vectors of mlx's mxfp4 format, for the CPU dequantization of the kernel cases.

    ref/.venv/bin/python ref/mxfp4_ref.py    prints packed codes, e8m0 scales and mx.dequantize(..., mode="mxfp4") in
                                             fp32 as Swift arrays

Two rows of 64 inputs (two groups of 32 each): the codes run through all 16 e2m1 patterns (both signs, -0 included) in
a fixed scramble, and the four scales span the range of gpt-oss-20b's experts (2^-11 ... 2^8). The values are copied
into Tests/MetalLabTests/KernelCaseMXFP4.swift (MXFP4ReferenceTests); the test checks that the CPU dequantization
(SafetensorsStore.dequantizeRows, fp4E2M1ToFloat, e8m0ToFloat) gives them, so the kernels are compared against what mlx
computes. It also checks here that mx.quantize(mode="mxfp4") of those values packs them back to the same words (value j
in bits 4j of each uint32, like the affine 4 bits).
"""
import mlx.core as mx

rows, cols = 2, 64
codes = [((7 * i + 3 * (i // 16)) % 16) for i in range(rows * cols)]
scales = [127 - 11, 127 + 8, 127 - 3, 127 + 2]  # one per group of 32, row-major
words = []
for r in range(rows):
    for w in range(cols // 8):
        v = 0
        for j in range(8):
            v |= codes[r * cols + 8 * w + j] << (4 * j)
        words.append(v)
w = mx.array(words, dtype=mx.uint32).reshape(rows, cols // 8)
s = mx.array(scales, dtype=mx.uint8).reshape(rows, cols // 32)
y = mx.dequantize(w, s, None, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32)
mx.eval(y)
wq, sq = mx.quantize(y, group_size=32, bits=4, mode="mxfp4")[:2]
mx.eval(wq, sq)
# -0 comes back as +0 from the quantizer; compare the dequantized values instead of the raw words for those.
y2 = mx.dequantize(wq, sq, None, group_size=32, bits=4, mode="mxfp4", dtype=mx.float32)
assert sq.tolist() == s.tolist(), "scales do not round-trip"
assert y2.tolist() == y.tolist(), "values do not round-trip"
same = sum(a == b for a, b in zip(wq.reshape(-1).tolist(), words))
print(f"// round trip: scales equal, values equal, {same}/{len(words)} words identical (-0 → +0 elsewhere)")
print("words:", "[" + ", ".join(f"0x{v:08X}" for v in words) + "]")
print("scales:", "[" + ", ".join(str(v) for v in scales) + "]")
print("values:", "[" + ", ".join(f"{v:.9g}" for v in y.reshape(-1).tolist()) + "]")
