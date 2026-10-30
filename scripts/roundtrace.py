#!/usr/bin/env python3
"""Round breakdown from the JSONL of `lab-dev gen --round-trace N --round-trace-out f.jsonl` (P0, fn-instr).

Each line is a round: encoders [label, layer, edge, t0, t1, value] in ms since gpuStartTime of the command buffer
and marks of the ExpertStream thread [value, layer, sees the signal, answers] on the same scale. Usage:

    roundtrace.py f.jsonl [--label 4K] [--skip 5]

For each round (mean over the rounds of the prompt): kernels per piece, gaps between encoders by edge type (signal,
CPU wait, diagnostic cut), pickup / CPU work / resume at the waits, and the post-wait slowdown test:
duration of fixed-work pieces right after a wait, against the median of the same piece in the round,
by how long the GPU sat idle.
"""
import json
import statistics as stt
import sys
from collections import defaultdict


def load(path, label, skip):
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if label:
        rows = [r for r in rows if r["label"] == label]
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    return {k: v[skip:] for k, v in by.items()}


def piece(label):
    # Level 1: labels of the natural encoders; level 2: pieces.
    return label


def analyse(rounds):
    n = len(rounds)
    kern = defaultdict(float)  # ms per round per piece
    gaps = defaultdict(float)  # ms per round per edge
    gpu = [r["gpu_ms"] for r in rounds]
    rows = [r["rows"] for r in rounds]
    nenc = []
    pick, work, resume, idle_wait, waits_cnt = [], [], [], [], 0
    h73 = defaultdict(list)  # bin -> ratios
    lab_round = defaultdict(lambda: defaultdict(list))
    missing = 0
    for r in rounds:
        encs = r["encs"]
        nenc.append(len(encs))
        cpu = {m[0]: m for m in r["cpu"]}
        prev_end = 0.0
        med = defaultdict(list)
        for e in encs:
            lab, _, edge, t0, t1, _ = e
            if t0 < 0:
                missing += 1
                continue
            med[lab].append(t1 - t0)
        medv = {k: stt.median(v) for k, v in med.items()}
        last_wait_idle = None
        since_wait = 99
        carry = None  # edge of an empty encoder (no marks): passes to the next one with marks
        for i, e in enumerate(encs):
            lab, layer, edge, t0, t1, val = e
            if t0 < 0:
                if carry is None or edge == "wait":
                    carry = (edge, val)
                continue
            if carry is not None:
                if carry[0] == "wait" or edge == "cut":
                    edge, val = carry
                carry = None
            kern[lab] += t1 - t0
            g = t0 - prev_end
            gaps["start" if i == 0 else edge] += g
            if edge == "wait":
                idle_wait.append(g)
                waits_cnt += 1
                last_wait_idle = g
                since_wait = 0
                m = cpu.get(val)
                if m is not None and i >= 2:
                    # End of the previous encoder to the signal of this value.
                    sig_i = next((j for j in range(i - 1, -1, -1) if encs[j][5] == val and encs[j][2] == "signal"), None)
                    if sig_i is not None and sig_i >= 1 and encs[sig_i - 1][4] >= 0:
                        tsig = encs[sig_i - 1][4]
                        pick.append(m[2] - tsig)
                        work.append(m[3] - m[2])
                        resume.append(t0 - max(m[3], prev_end))
            else:
                since_wait += 1
            # Post-wait slowdown: fixed-work pieces in the 1–6 encoders after a wait, against their median in the round.
            if last_wait_idle is not None and 1 <= since_wait <= 6 and lab in (
                "combine", "hc.a", "hc.f", "gdn", "attn", "qsa", "router", "split", "capture"
            ) and medv.get(lab, 0) > 0.005:
                b = "<0.05" if last_wait_idle < 0.05 else "0.05–0.2" if last_wait_idle < 0.2 else "0.2–0.5" if last_wait_idle < 0.5 else "≥0.5"
                ratio = (t1 - t0) / medv[lab]
                h73[b].append(ratio)
                lab_round[lab][b].append(ratio)
            prev_end = t1
        gaps["end"] += r["gpu_ms"] - prev_end
    out = {
        "n": n, "gpu_mean": stt.mean(gpu), "gpu_median": stt.median(gpu), "rows_mean": stt.mean(rows),
        "enc_mean": stt.mean(nenc), "kern": {k: v / n for k, v in kern.items()},
        "gaps": {k: v / n for k, v in gaps.items()}, "waits_per_round": waits_cnt / n,
        "pick": pick, "work": work, "resume": resume, "idle_wait": idle_wait, "h73": h73, "lab_round": lab_round,
        "missing": missing,
    }
    return out


def fmt(o, title):
    p = print
    p(f"## {title}: {o['n']} rounds, GPU mean {o['gpu_mean']:.2f} ms (median {o['gpu_median']:.2f}), "
      f"{o['rows_mean']:.2f} rows, {o['enc_mean']:.0f} encoders per round")
    if o["missing"]:
        p(f"  (encoders without marks: {o['missing']})")
    ks = sum(o["kern"].values())
    gs = sum(o["gaps"].values())
    p(f"  kernels (inside encoders) {ks:.2f} ms; gaps between encoders {gs:.2f} ms; sum {ks + gs:.2f}")
    p("  | piece | ms/round |")
    p("  |---|---|")
    for k, v in sorted(o["kern"].items(), key=lambda x: -x[1]):
        p(f"  | {k} | {v:.2f} |")
    p("  | **gap** | ms/round |")
    for k, v in sorted(o["gaps"].items(), key=lambda x: -x[1]):
        p(f"  | gap {k} | {v:.2f} |")
    if o["idle_wait"]:
        iw = o["idle_wait"]
        p(f"  waits: {o['waits_per_round']:.1f} per round; GPU idle per wait, mean {stt.mean(iw) * 1e3:.0f} µs "
          f"(median {stt.median(iw) * 1e3:.0f}); > 0.2 ms: {sum(1 for x in iw if x > 0.2) / max(1, o['n']):.1f} per round")
    if o["pick"]:
        def m(a):
            return f"mean {stt.mean(a) * 1e3:.0f} µs, median {stt.median(a) * 1e3:.0f}"
        p(f"  pickup (end of the encoder before the signal → the CPU sees it): {m(o['pick'])}")
        p(f"  CPU work (sees → answers): {m(o['work'])}")
        p(f"  resume (answers → the encoder after the wait starts, or from when the GPU is free): {m(o['resume'])}")
    if o["h73"]:
        p("  Post-wait slowdown: duration / median of the same piece in the round, fixed-work pieces 1–6 encoders after a wait, "
          "by the GPU idle time at that wait (ms):")
        for b in ("<0.05", "0.05–0.2", "0.2–0.5", "≥0.5"):
            a = o["h73"].get(b, [])
            if a:
                p(f"    {b}: n {len(a)}, mean {stt.mean(a):.3f}, median {stt.median(a):.3f}")
        for lab, d in sorted(o["lab_round"].items()):
            s = "; ".join(f"{b} {stt.mean(d[b]):.3f} (n {len(d[b])})" for b in ("<0.05", "0.05–0.2", "0.2–0.5", "≥0.5") if d.get(b))
            p(f"    {lab}: {s}")


def main():
    a = sys.argv[1:]
    path = a[0]
    label = a[a.index("--label") + 1] if "--label" in a else None
    skip = int(a[a.index("--skip") + 1]) if "--skip" in a else 5
    for k, v in load(path, label, skip).items():
        if v:
            fmt(analyse(v), f"{path.split('/')[-1]} {k}")


if __name__ == "__main__":
    main()
