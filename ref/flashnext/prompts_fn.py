"""Benchmark prompts (as in ref/prompts.py) with the Flash-Next chat template, for lab-dev gen.

The Flash-Next template (chat_template.jinja) adds a system message with the reasoning instructions
(reasoning_effort, xhigh by default) and opens the <think> block. It is written by hand (no transformers or jinja) for a
single user message and its shape is checked with a fixed case. Writes ref/out/prompts-fn/<name>.json.

Usage: python -I prompts_fn.py --model <model-dir> [--seed 1] [--effort xhigh|medium|low]
"""
import argparse
import glob
import json
import os
import random
import sys
import uuid

from tokenizers import Tokenizer

TARGETS = {"512": 1600, "4K": 13000, "16K": 52000, "32K": 104000}
QUESTION = ("\n\nBased on the code above, explain how the event loop schedules callbacks, "
            "then write a small example using asyncio.gather with error handling.")
CHAT = "Explica en pocas frases qué es la decodificación especulativa y por qué el resultado es idéntico al normal."
EFFORT = {
    'xhigh': 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, '
             'consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.',
    'medium': '',
    'low': 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion '
           'without unnecessary elaboration.',
}


def render(user, effort):
    sys_ = EFFORT[effort]
    s = f"<|im_start|>system\n{sys_}<|im_end|>\n" if sys_ else ""
    return s + f"<|im_start|>user\n{user.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/Models/Qwen3.8-Flash-Next-lab-Q2"))
    ap.add_argument("--asyncio-dir", default=os.path.expanduser(
        "~/.local/share/mise/installs/python/3.14/lib/python3.14/asyncio"))
    ap.add_argument("--seed", default="1")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--text-dir", default=None, help="with --es: folder with the Spanish markdown pages of prompts_es.py")
    ap.add_argument("--es", action="store_true",
                    help="the prompts of ref/prompts_es.py (chat and code in several languages, 4K text) in prompts-fn-es")
    a = ap.parse_args()
    if a.es:
        return main_es(a)
    tok = Tokenizer.from_file(os.path.join(os.path.expanduser(a.model), "tokenizer.json"))
    corpus = "".join(open(f).read() for f in sorted(glob.glob(a.asyncio_dir + "/*.py")))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "prompts-fn")
    os.makedirs(out, exist_ok=True)
    n = 1
    prompts = {"chat": CHAT}
    for name, chars in TARGETS.items():
        n += 1
        sid = uuid.UUID(int=random.Random(f"{a.seed}-{n}").getrandbits(128), version=4)
        prompts[name] = f"[session {sid}]\n" + corpus[:chars] + QUESTION
    for name, text in prompts.items():
        ids = tok.encode(render(text, a.effort), add_special_tokens=False).ids
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump({"name": name, "chars": len(text), "ids": list(map(int, ids))}, f)
        print(f"{name}: {len(ids)} tokens; final: {tok.decode(ids[-8:])!r}")


def main_es(a):
    """The texts of ref/prompts_es.py (its constants, without importing transformers) with the Flash-Next template."""
    import ast
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts_es.py")).read()
    k = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                "PAGES", "TARGETS", "QUESTION", "CODE", "CHAT", "MULTI"):
            k[node.targets[0].id] = ast.literal_eval(node.value)
    tok = Tokenizer.from_file(os.path.join(os.path.expanduser(a.model), "tokenizer.json"))
    if not a.text_dir:
        sys.exit("--es needs --text-dir: the folder with the Spanish markdown pages named in prompts_es.py PAGES")
    corpus = "\n\n".join(open(os.path.join(os.path.expanduser(a.text_dir), p)).read() for p in k["PAGES"])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "prompts-fn-es")
    os.makedirs(out, exist_ok=True)
    prompts = {"chat": k["CHAT"], "es-chat": k["CHAT"], "es-code": k["CODE"]}
    prompts.update(k["MULTI"])
    prompts["es-4K"] = corpus[:k["TARGETS"]["es-4K"]] + k["QUESTION"]
    for name, text in prompts.items():
        ids = tok.encode(render(text, a.effort), add_special_tokens=False).ids
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump({"name": name, "chars": len(text), "ids": list(map(int, ids))}, f)
        print(f"{name}: {len(ids)} tokens")


if __name__ == "__main__":
    main()
