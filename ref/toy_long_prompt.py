"""Long prompt for the toy models: the wrapper of toyPrompts (Sources/ingot-dev/Toy/ToyModels.swift: user
header, body, tail with <think>) tokenized with a toy's tokenizer.json (the same in all of them), with the three
ref/agent_dump/*.json concatenated as the body (0008 first, so the prefix is the one of the 4K prompt). Writes
out/prompts-toy-long/<name>.json ({"ids": [...]}), outside prompts-toy so the toys and toy-shapes do not change.

--check rebuilds 512 and 4K from 0008 alone and compares them with out/prompts-toy (same tokenizer and wrapper).

Usage:
  .venv/bin/python toy_long_prompt.py --toy ~/.cache/ingot/toy/toy-granite --tokens 15989 --name 16K [--check]
"""
import argparse
import json
import os
import re

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD, TAIL = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n<think>\n"
DUMPS = ("0008-body.json", "0001-body.json", "0015-body.json")


def encoder(path: str):
    """Text -> ids. The special tokens take the ids that tokenizer.json declares in added_tokens, as ingot's
    BPETokenizer does: the toy vocabulary is cut and the tokenizers library renumbers added tokens that overlap it."""
    tok = Tokenizer.from_file(path)
    special = {a["content"]: a["id"] for a in json.load(open(path))["added_tokens"]}
    split = re.compile("(" + "|".join(re.escape(t) for t in sorted(special, key=len, reverse=True)) + ")")

    def enc(text: str) -> list:
        ids = []
        for part in split.split(text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids += tok.encode(part, add_special_tokens=False).ids
        return ids

    return enc


def build(enc, text: str, n: int) -> list:
    head, tail, body = enc(HEAD), enc(TAIL), enc(text)
    m = n - len(head) - len(tail)
    if len(body) < m:
        print(f"the body has {len(body)} tokens, short of {m}: the prompt is {len(head) + len(body) + len(tail)} tokens")
        m = len(body)
    return head + body[:m] + tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", required=True, help="toy model directory (its tokenizer.json)")
    ap.add_argument("--tokens", type=int, default=15989, help="prompt length (not a multiple of 64, like 509 and 3989)")
    ap.add_argument("--name", default="16K")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    tok = encoder(os.path.join(os.path.expanduser(a.toy), "tokenizer.json"))
    read = lambda f: open(os.path.join(HERE, "agent_dump", f), encoding="utf-8").read()  # noqa: E731
    if a.check:
        for name, n in (("512", 509), ("4K", 3989)):
            ref = json.load(open(os.path.join(HERE, "out", "prompts-toy", f"{name}.json")))["ids"]
            ok = build(tok, read(DUMPS[0]), n) == ref
            print(f"check {name}: {'same ids as' if ok else 'DIFFERENT from'} out/prompts-toy/{name}.json")
            assert ok
    ids = build(tok, "\n".join(read(f) for f in DUMPS), a.tokens)
    d = os.path.join(HERE, "out", "prompts-toy-long")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{a.name}.json"), "w") as f:
        json.dump({"ids": ids}, f)
    print(f"out/prompts-toy-long/{a.name}.json: {len(ids)} tokens")


if __name__ == "__main__":
    main()
