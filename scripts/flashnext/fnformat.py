"""ingot format for Qwen3.8-Flash-Next (qwen4_exp_text) and a minimal GGUF reader.

The engine never reads GGUF: this module is only used by the converter (once) to extract the ISTA Q2_0 GSQ experts.

experts.bin: one record per expert, [layer][expert], of RECORD bytes (85 pages of 16 KiB). Inside, planes:
  gate codes [640, 2560/16] u32 | gate scales [640, 40] f16 | up (same) | down codes [2560, 640/16] u32 |
  down scales [2560, 10] f16 | zero padding.
2-bit codes in MLX order (value j of each little-endian u32 in bits 2j), fp16 scale per group of
64 without bias: w = d · (q − 1). It is the GGUF block_q2_0 (half d | 16 B of codes, element l in bits
2·(l mod 4) of qs[l/4]) with the planes separated: the codes are copied as they are and the scale stays fp16.

ple.bin: the n-gram table in MLX 4 bits g32 (from mlx-community, not requantized), in 16 KiB pages with 163
rows of 100 B each (codes 80 B = 20 u32 | 5 bf16 scales | 5 bf16 biases); row r is in page r // 163,
offset 100 · (r % 163). No row crosses a page.
"""
import struct
import numpy as np

HIDDEN, INTER, EXPERTS, LAYERS, GROUP = 2560, 640, 512, 48, 64
PAGE = 16384


def plane_sizes(n, k):
    """Bytes of codes and of scales of an [n, k] matrix in planar Q2_0."""
    return n * k // 4, n * (k // GROUP) * 2


GATE_C, GATE_S = plane_sizes(INTER, HIDDEN)
DOWN_C, DOWN_S = plane_sizes(HIDDEN, INTER)
OFFSETS = {}
_o = 0
for _name, (_c, _s) in (("gate", (GATE_C, GATE_S)), ("up", (GATE_C, GATE_S)), ("down", (DOWN_C, DOWN_S))):
    OFFSETS[_name] = (_o, _o + _c)
    _o += _c + _s
PAYLOAD = _o  # 1,382,400
RECORD = (PAYLOAD + PAGE - 1) // PAGE * PAGE  # 1,392,640

PLE_ROW, PLE_ROWS_PER_PAGE, PLE_DIM, PLE_GROUP = 100, 163, 160, 32

# ---------------------------------------------------------------- GGUF (header + offsets only)
_KV = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
GGML_Q2_0 = 42


def _rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _rs(f):
    return f.read(_rd(f, '<Q')).decode('utf-8', 'replace')


def _rv(f, t):
    if t == 8:
        return _rs(f)
    if t == 9:
        at, n = _rd(f, '<I'), _rd(f, '<Q')
        return [_rv(f, at) for _ in range(n)]
    return _rd(f, _KV[t])


def read_gguf(path):
    """(metadata, {name: (dims, type, absolute offset)})."""
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        _rd(f, '<I')
        nt, nk = _rd(f, '<Q'), _rd(f, '<Q')
        meta = {}
        for _ in range(nk):
            k = _rs(f)
            t = _rd(f, '<I')
            meta[k] = _rv(f, t)
        ts = {}
        for _ in range(nt):
            name = _rs(f)
            nd = _rd(f, '<I')
            dims = [_rd(f, '<Q') for _ in range(nd)]
            t = _rd(f, '<I')
            ts[name] = (dims, t, _rd(f, '<Q'))
        align = meta.get('general.alignment', 32)
        base = (f.tell() + align - 1) // align * align
    return meta, {k: (d, t, base + o) for k, (d, t, o) in ts.items()}


# ---------------------------------------------------------------- Q2_0
def q2_blocks_to_planes(raw, n, k):
    """block_q2_0 [n, k/64] (18 B) → (codes [n, k/4] u8, scales [n, k/64] f16): byte copy, no arithmetic."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, k // GROUP, 18)
    return np.ascontiguousarray(b[:, :, 2:]).reshape(n, k // 4), np.ascontiguousarray(b[:, :, :2]).view(np.float16)[..., 0]


def dequant_gguf_q2(raw, n, k):
    """Reference dequantization of block_q2_0 (byte by byte, like gguf_embedding_formats.h of splash)."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n, k // GROUP, 18)
    d = b[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32)
    qs = b[:, :, 2:]  # [n, g, 16]
    l = np.arange(GROUP)
    q = (qs[:, :, l // 4] >> (2 * (l % 4)).astype(np.uint8)) & 3
    return ((q.astype(np.float32) - 1.0) * d[..., None]).reshape(n, k)


def dequant_planar_q2(codes, scales, n, k):
    """Dequantization of our format with the MLX reading (little-endian u32, value j in bits 2j)."""
    w = np.frombuffer(codes.tobytes(), dtype='<u4').reshape(n, k // 16)
    j = np.arange(16, dtype=np.uint32)
    q = ((w[..., None] >> (2 * j)) & 3).reshape(n, k).astype(np.float32)
    s = np.frombuffer(scales.tobytes(), dtype=np.float16).reshape(n, k // GROUP).astype(np.float32)
    return (q - 1.0) * np.repeat(s, GROUP, axis=1)


def record_planes(rec):
    """View of a record: {'gate': (codes, scales), ...} as byte arrays."""
    rec = np.frombuffer(rec, dtype=np.uint8)
    shapes = {"gate": (INTER, HIDDEN), "up": (INTER, HIDDEN), "down": (HIDDEN, INTER)}
    out = {}
    for name, (c0, s0) in OFFSETS.items():
        n, k = shapes[name]
        cb, sb = plane_sizes(n, k)
        out[name] = (rec[c0:c0 + cb], rec[s0:s0 + sb], n, k)
    return out
