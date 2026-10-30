"""Reference for the full pre_tokenizer (a Sequence of one or more Split steps and ByteLevel) and for the merge order of
tokenizers, for LabServe. Writes Tests/LabServeTests/Resources/multisplit/{tokenizer.json,cases.json}.

tokenizer.json is a tiny byte-level BPE (trained here with tokenizers) with the pre_tokenizer of Qwen3.6
(froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit: one Split and ByteLevel without regex) and, ahead of the other
merges, the newline merges in LFM2.5 order (a freshly formed merge may rank below the one that formed it). cases.json
gives the tokenizers ids for that pre_tokenizer and for variants: several Split steps in a row (hand written: digits,
CJK runs, then gpt-oss's word expression), the other behaviors, invert, String patterns (Gemma's MergedWithPrevious on
" ") and ByteLevel with prefix and expression (Granite). Variants that tokenizers does not load, or that LabServe does
not support (Digits, which no tier 1 family uses), go in "errors".

Usage: uv run --no-project --with tokenizers==0.23.3 python -I ref/tok_multisplit_ref.py
"""
import copy
import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tests", "LabServeTests", "Resources", "multisplit")
# Qwen3.6's Split expression and gpt-oss's (o200k) word expression, from their tokenizer.json.
QWEN36 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
O200K_WORDS = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
               r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
               r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def split(rx=None, s=None, behavior="Isolated", invert=False):
    return {"type": "Split", "pattern": {"Regex": rx} if rx is not None else {"String": s}, "behavior": behavior,
            "invert": invert}


def byte_level(prefix=False, regex=False):
    return {"type": "ByteLevel", "add_prefix_space": prefix, "trim_offsets": True, "use_regex": regex}


def seq(*steps):
    return {"type": "Sequence", "pretokenizers": list(steps)}


QWEN36_PT = seq(split(QWEN36), byte_level())
VARIANTS = {
    "qwen36": QWEN36_PT,
    "three-splits": seq(split(r"\p{N}{1,3}"), split("[一-龥぀-ゟ゠-ヿ]+"), split(O200K_WORDS), byte_level()),
    "merged-next-string": seq(split(s=" ", behavior="MergedWithNext"), byte_level()),
    "merged-previous-string": seq(split(s=" ", behavior="MergedWithPrevious"), byte_level()),
    "merged-previous-invert": seq(split(r"[\p{P}\p{S}]", behavior="MergedWithPrevious", invert=True), byte_level()),
    "contiguous-removed": seq(split(r"[a-z]", behavior="Contiguous"), split(r"\p{P}", behavior="Removed"), byte_level()),
    "bytelevel-prefix-regex": byte_level(prefix=True, regex=True),
}
ERRORS = {
    "whitespace": seq({"type": "Whitespace"}, byte_level()),
    "digits": seq({"type": "Digits", "individual_digits": True}, byte_level(prefix=True, regex=True)),
    "bytelevel-not-last": seq(byte_level(), split(r"\p{N}")),
    "no-bytelevel": seq(split(r"\p{N}")),
}
TEXTS = [
    "Hola, ¿qué tal? Él cogió el camión en Logroño.",
    "1234567 + 89 = 1234656; 3.14159 y ١٢٣٤ y ½",
    "价格是123元，日期2026年10月21日。カタカナ123ひらがな한국어",
    "CamelCaseWord HTTPServer getHTTPResponse I'm DON'T we'll",
    "def f(x):\n    return x**2  # cuadrado\n\n\n\tpass\r\n",
    "\n" * 30 + "a" + "\n" * 13 + " " * 7,
    "😀👍🏽 ‍ zwj  é Å \x01 fin",
    "<think>razono</think><tool_call>{\"a\": 1}</tool_call>",
]
TRAIN = TEXTS + ["El modelo local tokeniza texto en español, inglés, código y chino.",
                 "The quick brown fox jumps over the lazy dog. 中文分词测试。", "for i in range(10):\n    print(i)\n"] * 20


def build():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(TRAIN, trainers.BpeTrainer(
        vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    d = json.loads(tok.to_str())
    vocab, merges = d["model"]["vocab"], d["model"]["merges"]
    nl = []  # LFM2.5's newline merges: Ċ^9+Ċ, Ċ^8+Ċ^2, ..., Ċ+Ċ^9, Ċ^8+Ċ, ..., Ċ+Ċ
    for total in range(10, 1, -1):
        nl += [["Ċ" * a, "Ċ" * (total - a)] for a in range(total - 1, 0, -1)]
    merges = nl + [m for m in merges if set("".join(m)) != {"Ċ"}]
    for k in range(2, 11):
        vocab.setdefault("Ċ" * k, len(vocab))
    d["model"]["merges"], d["model"]["vocab"] = merges, vocab
    d["added_tokens"] = [{"id": len(vocab) + i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": i < 2}
                         for i, c in enumerate(["<think>", "</think>", "<tool_call>", "</tool_call>"])]
    d["normalizer"], d["pre_tokenizer"] = None, QWEN36_PT
    d["decoder"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    return d


def main():
    d = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    cases = {"variants": [], "errors": list(ERRORS.values())}
    for name, pt in VARIANTS.items():
        v = copy.deepcopy(d)
        v["pre_tokenizer"] = pt
        tok = Tokenizer.from_str(json.dumps(v))
        cases["variants"].append({"name": name, "pre_tokenizer": pt, "cases": [
            {"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXTS]})
    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print({v["name"]: sum(len(c["ids"]) for c in v["cases"]) for v in cases["variants"]})


if __name__ == "__main__":
    main()
