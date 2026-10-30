"""Flash-Next PLE table into ple.bin (fnformat.py), from mlx-community/Qwen3.8-Flash-Next-4bit (MLX 4 bits g32,
not requantized). Fetches by HTTP ranges only the 3 × 128 tensors of the table, chunk by chunk, and does not keep the download.

Usage: python -I convert_ple.py <folder with the .json headers of the repo safetensors> <output> [threads]
The headers are fetched by st_headers.py (or any header reader). Resumable by chunks.
Checks each chunk by rereading 4096 random rows of ple.bin against what was downloaded.
"""
import os
import sys
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fnformat as F

REPO, REV = 'mlx-community/Qwen3.8-Flash-Next-4bit', '07b5dc6c54600a359b87f1e53e7adf6351c72a2c'
hdrdir, out = sys.argv[1], sys.argv[2]
nth = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ROWS, NSH = 320001536, 128
PER = ROWS // NSH
NPAGES = (ROWS + F.PLE_ROWS_PER_PAGE - 1) // F.PLE_ROWS_PER_PAGE
hdr = build_hf_headers()
loc = {}
for hp in os.listdir(hdrdir):
    h = json.load(open(os.path.join(hdrdir, hp)))
    n = h.pop('__header_len__')
    for k, v in h.items():
        m = re.search(r'ngram_embedding\.shards\.(\d+)\.(weight|scales|biases)$', k)
        if m:
            loc[(int(m.group(1)), m.group(2))] = (hp[:-5], 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
assert len(loc) == 3 * NSH, len(loc)

def fetch(f, a, sz):
    url = hf_hub_url(REPO, f, revision=REV)
    CH = 16 << 20
    buf = bytearray(sz)
    def one(c):
        for i in range(8):
            try:
                h = dict(hdr); h['Range'] = f'bytes={a + c}-{a + min(sz, c + CH) - 1}'
                r = get_session().get(url, headers=h, follow_redirects=True, timeout=120)
                r.raise_for_status()
                assert len(r.content) == min(CH, sz - c)
                buf[c:c + len(r.content)] = r.content
                return
            except Exception as e:
                print('retry', f, c, repr(e)[:120], flush=True); time.sleep(2 + 3 * i)
        raise RuntimeError('download')
    with ThreadPoolExecutor(nth) as ex:
        list(ex.map(one, range(0, sz, CH)))
    return bytes(buf)

os.makedirs(out, exist_ok=True)
dst = os.path.join(out, 'ple.bin'); prog = dst + '.prog'
done = set(json.load(open(prog))) if os.path.exists(prog) else set()
fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
if os.fstat(fd).st_size != NPAGES * F.PAGE:
    os.ftruncate(fd, NPAGES * F.PAGE)
rng = np.random.default_rng(1)
t0 = time.time(); got = 0
for s in range(NSH):
    if s in done:
        continue
    parts = {}
    for kind, dt, cols in (('weight', 'U32', 20), ('scales', 'BF16', 5), ('biases', 'BF16', 5)):
        f, a, sz, dtype, shape = loc[(s, kind)]
        assert dtype == dt and shape == [PER, cols], (s, kind, dtype, shape)
        parts[kind] = np.frombuffer(fetch(f, a, sz), dtype=np.uint8).reshape(PER, cols * (4 if dt == 'U32' else 2))
        got += sz
    rows = np.concatenate([parts['weight'], parts['scales'], parts['biases']], axis=1)  # [PER, 100]
    r0 = s * PER
    p0, p1 = r0 // F.PLE_ROWS_PER_PAGE, (r0 + PER - 1) // F.PLE_ROWS_PER_PAGE + 1
    pages = np.frombuffer(bytearray(os.pread(fd, (p1 - p0) * F.PAGE, p0 * F.PAGE)), dtype=np.uint8).reshape(p1 - p0, F.PAGE)
    W = F.PLE_ROWS_PER_PAGE * F.PLE_ROW
    slots = pages[:, :W].reshape(-1, F.PLE_ROW).copy()  # (the view is not contiguous: copy and write back)
    a0 = r0 - p0 * F.PLE_ROWS_PER_PAGE
    slots[a0:a0 + PER] = rows
    pages[:, :W] = slots.reshape(p1 - p0, W)
    os.pwrite(fd, pages.tobytes(), p0 * F.PAGE)
    os.fsync(fd)
    # check: 4096 random rows reread from ple.bin
    for r in rng.choice(PER, 4096, replace=False):
        g = r0 + int(r)
        b = os.pread(fd, F.PLE_ROW, (g // F.PLE_ROWS_PER_PAGE) * F.PAGE + (g % F.PLE_ROWS_PER_PAGE) * F.PLE_ROW)
        assert b == rows[r].tobytes(), (s, r)
    done.add(s)
    json.dump(sorted(done), open(prog, 'w'))
    el = time.time() - t0
    print(f'chunk {s}: {got / 1e9:.2f} GB in {el:.0f} s ({got / el / 1e6:.0f} MB/s)', flush=True)
os.close(fd)
if len(done) == NSH and os.path.exists(prog):
    os.remove(prog)
print('DONE', NPAGES * F.PAGE, 'bytes', flush=True)
