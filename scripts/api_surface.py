#!/usr/bin/env python3
"""Engine API surface, read from the sources (no build, no GPU), to review its diffs.

    scripts/api_surface.py            writes ref/api-surface.txt
    scripts/api_surface.py --check    exits with 1 if ref/api-surface.txt does not match (used by scripts/ci.sh)
    scripts/api_surface.py --exp-check  exits with 1 if Sources/ reads `exp` (EngineOptions.exp) outside EngineOptions.swift

Contents: subcommands and options of `lab` and `lab-dev`, EngineOptions fields, the HTTP request fields that
`lab serve` reads (ChatServer.swift; the ones it rejects when not neutral, marked), environment variables read in
Sources/ (each with its reason: a variable without a reason in ENV_REASONS fails) and the tuning options kept on
purpose (PERMANENT, each with when to use it; one that no binary reads any more fails). All sorted.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Sources"
OUT = ROOT / "ref" / "api-surface.txt"

# Environment variables allowed in Sources/ and why (any others may not be read).
ENV_REASONS = {
    "HF_HUB_CACHE": "Hugging Face cache path (huggingface_hub convention; ModelPaths)",
    "METAL_LAB_MODELS": "directory of models for `lab plan` without --model (no models path is built into the code)",
    "METAL_LAB_SKIP_CHECK": "skips the machine requirements check (MachineRequirements; for tests only)",
}

# Tuning and diagnostic options that stay on purpose (not experiments awaiting removal), and when to use them.
PERMANENT = {
    "agent-guards": "lab serve; on to stop loops in agent sessions (whoever starts lab serve for pi asks for it)",
    "expert-fn-bulk": "Flash-Next prefill: re-tune the bulk-read threshold when measuring prefill (SSD, cache size)",
    "expert-pf-bulk": "MoE prefill with SSD experts: re-tune the bulk-read threshold when measuring prefill",
    "expert-pf-group": "SSD-expert prefill: chunks per group; 1 to compare chunk by chunk, or re-tune for memory",
    "expert-prefetch": "SSD experts: read-ahead top-K; 0 to measure without read-ahead, or re-tune per model and cache",
    "expert-wide": "Flash-Next: wide read-ahead width; re-tune with a new cache size, SSD or prediction; 0 = no wide",
    "mincore": "lab-dev fncheck: page-cache residency of the expert and PLE files, to check SSD reads are real",
}

ACCESSORS = {"opts": "value", "flags.contains": "flag", "int": "value", "str": "value", "required": "required"}
ACC_RE = re.compile(r'args\.(opts\[|flags\.contains\(|int\(|str\(|required\()\s*"([^"]+)"')


def swift_files(*dirs):
    for d in dirs:
        yield from sorted((SRC / d).rglob("*.swift"))


def options(*dirs):
    found = {}
    for f in swift_files(*dirs):
        for m in ACC_RE.finditer(f.read_text()):
            kind = {"opts[": "opts", "flags.contains(": "flags.contains", "int(": "int", "str(": "str",
                    "required(": "required"}[m.group(1)]
            found.setdefault(m.group(2), set()).add(ACCESSORS[kind])
    return found


def subcommands(binary):
    out, parent = [], None
    for line in (SRC / binary / "main.swift").read_text().splitlines():
        m = re.match(r'^( {4}| {8})case ((?:"[^"]+"(?:, )?)+):', line)
        if not m:
            continue
        names = re.findall(r'"([^"]+)"', m.group(2))
        if len(m.group(1)) == 4:
            parent = names[0]
            out += names
        elif parent == "kl":
            out += [f"{parent} {n}" for n in names]
    return sorted(set(out))


def engine_fields():
    lines, stack, out = (SRC / "MetalLab" / "EngineOptions.swift").read_text().splitlines(), [], []
    for line in lines:
        s = line.strip()
        m = re.match(r"public (?:struct|enum) (\w+)", s)
        if m:
            stack.append((m.group(1), len(line) - len(line.lstrip())))
            continue
        if stack and s == "}" and len(line) - len(line.lstrip()) == stack[-1][1]:
            stack.pop()
            continue
        m = re.match(r"public var (\w+)\s*(?::\s*([^={]+?))?\s*(?:=\s*(.+?))?(?:\s*\{)?$", s)
        if m and stack and not s.endswith("{") and ("= " in s or ":" in s):
            owner = stack[-1][0]
            if owner == "EngineOptions" or owner.endswith("Options"):
                typ = (m.group(2) or "").strip()
                dflt = (m.group(3) or "").strip()
                out.append(f"{owner}.{m.group(1)}" + (f": {typ}" if typ else "") + (f" = {dflt}" if dflt else ""))
    return sorted(out)


def http_fields():
    """Top-level request fields read by ChatServer (`b["key"]`, `b["key"]?["sub"]`, `num("key")`), and the ones that
    rejectUnsupported answers with a 400 when not neutral."""
    text = (SRC / "LabServe" / "ChatServer.swift").read_text()
    read = set()
    for m in re.finditer(r'\bb\["([^"]+)"\](?:\?\["([^"]+)"\])?', text):
        read.add(m.group(1) + (f".{m.group(2)}" if m.group(2) else ""))
    read |= set(re.findall(r'\bnum\("([^"]+)"\)', text))
    m = re.search(r"static func rejectUnsupported\(.*?\n    \}\n", text, re.S)
    rejected = set()
    if m:
        body = m.group(0)
        rejected |= set(re.findall(r'\bb\["([^"]+)"\]', body))
        for lst in re.findall(r"for k in \[([^\]]+)\]", body):
            rejected |= set(re.findall(r'"([^"]+)"', lst))
    return sorted(read - rejected), sorted(rejected)


def env_reads():
    found = {}
    for f in sorted(SRC.rglob("*.swift")):
        for m in re.finditer(r'environment\[\s*"([^"]+)"\s*\]', f.read_text()):
            found.setdefault(m.group(1), set()).add(str(f.relative_to(ROOT)))
    return found


def exp_uses():
    """Uses of the exp API in Sources/ outside its definition."""
    bad = []
    pat = re.compile(r"\.exp\b|\bexp\(\s*\"")
    for f in sorted(SRC.rglob("*.swift")):
        if f.name in ("EngineOptions.swift", "Args.swift"):  # Args: parsing of --exp
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("//")[0]
            # `o.exp = args.exp` (the CLI that fills it) is the only assignment allowed.
            if pat.search(code) and "o.exp = args.exp" not in code and "var exp" not in code:
                bad.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    return bad


def render():
    common = options("LabCLI")
    common["exp"] = {"value key=value, repeatable: EngineOptions.exp"}  # parsed by Args, not an accessor
    L = ["# Engine API surface. Generated by scripts/api_surface.py: do not edit by hand.",
         "# Regenerate: scripts/api_surface.py   (check: scripts/api_surface.py --check)", ""]

    def opt_lines(d, title):
        L.append(title)
        for k in sorted(d):
            L.append(f"  --{k}  [{', '.join(sorted(d[k]))}]")
        L.append("")

    for binary in ("lab", "lab-dev"):
        L.append(f"## {binary}: subcommands")
        L += [f"  {c}" for c in subcommands(binary)]
        L.append("")
        own = {k: v for k, v in options(binary).items() if k not in common}
        opt_lines(own, f"## {binary}: own options")
    opt_lines(common, "## Common options (LabCLI: read by lab and lab-dev)")
    L.append("## EngineOptions: fields (name: type = default value)")
    L += [f"  {x}" for x in engine_fields()]
    L.append("")
    accepted, rejected = http_fields()
    L.append("## lab serve: request fields (POST /v1/chat/completions; LabServe/ChatServer.swift)")
    L += [f"  {k}" for k in accepted]
    L += [f"  {k}  [400 if not neutral]" for k in rejected]
    L.append("")
    L.append("## Permanent tuning options (kept on purpose; when to use them: PERMANENT in scripts/api_surface.py)")
    L += [f"  --{k}  {PERMANENT[k]}" for k in sorted(PERMANENT)]
    L.append("")
    L.append("## Environment variables read in Sources/ (and why)")
    reads = env_reads()
    for v in sorted(reads):
        L.append(f"  {v}  {ENV_REASONS.get(v, '??? NO REASON: remove it or add it to ENV_REASONS in scripts/api_surface.py')}")
        L.append(f"      read in: {', '.join(sorted(reads[v]))}")
    L.append("")
    return "\n".join(L)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "--exp-check":
        bad = exp_uses()
        if bad:
            print("FAIL: Sources/ uses the exp API (EngineOptions.exp) outside EngineOptions.swift; no reads may "
                  "remain in main:\n  " + "\n  ".join(bad), file=sys.stderr)
            sys.exit(1)
        return
    text = render()
    read = set(options("LabCLI")) | set(options("lab")) | set(options("lab-dev"))
    gone = sorted(k for k in PERMANENT if k not in read)
    if gone:
        print(f"FAIL: PERMANENT in scripts/api_surface.py names options no binary reads: {', '.join(gone)}",
              file=sys.stderr)
        sys.exit(1)
    unknown = [v for v in env_reads() if v not in ENV_REASONS]
    if unknown:
        print(f"FAIL: environment variables without an allowed reason: {', '.join(unknown)} (Sources/ does not read "
              "environment variables except those in ENV_REASONS in scripts/api_surface.py)", file=sys.stderr)
        sys.exit(1)
    if mode == "--check":
        if not OUT.exists() or OUT.read_text() != text:
            print("FAIL: ref/api-surface.txt does not match the sources. Regenerate with scripts/api_surface.py and "
                  "review the diff (git diff ref/api-surface.txt).", file=sys.stderr)
            sys.exit(1)
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


main()
