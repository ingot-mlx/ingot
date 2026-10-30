"""Reference for the SentencePiece-style BPE of IngotServe (Gemma 3 and 4): a tiny tokenizer.json trained here with
tokenizers and set up like Gemma's (Replace(" "→"\u2581") normalizer, Split(" ", MergedWithPrevious), byte_fallback with
the 256 <0xNN> tokens, unk with fuse_unk, Replace(\u2581→" ")+ByteFallback+Fuse decoder, non-normalized added tokens).
Writes Tests/IngotServeTests/Resources/spm/{tokenizer.json,cases.json}.

cases.json has, for that tokenizer and for variants (Prepend("\u2581") normalizer; no byte_fallback with unk fused, not
fused, or no unk at all), the ids and decode of tokenizers for each text, and decode cases on hand-picked ids (runs of
byte tokens that are not valid UTF-8, added tokens with \u2581). Variants IngotServe must reject go in "errors".

Usage: uv run --no-project --with tokenizers==0.23.3 python -I ref/tok_spm_ref.py
"""
import copy
import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tests", "IngotServeTests", "Resources", "spm")
TEXTS = [
    "Hola, ¿qué tal? Él cogió el camión en Logroño.",
    "  leading and trailing  ",
    "def f(x):\n    return x**2  # square\n\n\n\tpass\r\n",
    "价格是123元，日期2026年10月21日。カタカナ한국어",
    "😀👍🏽 \u200d zwj é Å \x01 \x00 \ue000 \U00020000 end",
    "\u2581\u2581 literal sentencepiece marks \u2581 and <b>bold</b> <start_of_turn>user\nhi<end_of_turn>",
    "Mixed ¿¡«» “quotes” — dash… €£¥ ∑∫√",
    "",
    " ",
    "\n\n\n",
]
TRAIN = TEXTS[:4] + ["El modelo local tokeniza texto en español, inglés, código y chino.",
                     "The quick brown fox jumps over the lazy dog.", "for i in range(10):\n    print(i)\n"] * 20
ADDED = [("<start_of_turn>", True), ("<end_of_turn>", True), ("<b>", False), ("</b>", False), ("\u2581\u2581", False),
         ("\n\n", False)]


def build():
    tok = Tokenizer(models.BPE(unk_token="<unk>", fuse_unk=True, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.train_from_iterator(TRAIN, trainers.BpeTrainer(vocab_size=500, special_tokens=["<pad>", "<unk>"]))
    d = json.loads(tok.to_str())
    vocab = d["model"]["vocab"]
    for b in range(256):
        vocab.setdefault(f"<0x{b:02X}>", len(vocab))
    for c, _ in ADDED:
        vocab.setdefault(c, len(vocab))  # Gemma's added tokens are also in the model vocabulary
    d["added_tokens"] = [{"id": vocab[c], "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": sp} for c, sp in ADDED]
    d["normalizer"] = {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}
    d["pre_tokenizer"] = {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious",
                          "invert": False}
    d["decoder"] = {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]}
    d["post_processor"] = None
    return d


def variants(d):
    out = {"gemma": d}
    v = copy.deepcopy(d)
    v["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "\u2581"}, d["normalizer"]]}
    out["prepend"] = v
    for name, fuse, unk in [("unk-fused", True, "<unk>"), ("unk-not-fused", False, "<unk>"), ("no-unk", True, None)]:
        v = copy.deepcopy(d)
        v["model"]["byte_fallback"], v["model"]["fuse_unk"], v["model"]["unk_token"] = False, fuse, unk
        out[name] = v
    return out


def errors(d):
    out = []
    v = copy.deepcopy(d)
    v["decoder"]["decoders"].append({"type": "Strip", "content": " ", "start": 1, "stop": 0})
    out.append(v)
    v = copy.deepcopy(d)
    v["added_tokens"][2]["normalized"] = True
    out.append(v)
    v = copy.deepcopy(d)
    v["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    out.append(v)
    return out


def main():
    d = build()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
    vocab = d["model"]["vocab"]
    b = lambda x: vocab[f"<0x{x:02X}>"]
    decode_ids = [[b(0xE4), b(0xBD), b(0xA0)], [b(0xE4), b(0xBD), b(0xA0), b(0xFF)], [b(0xE4), vocab["a"], b(0xBD)],
                  [vocab["\u2581\u2581"], vocab["a"], vocab["<start_of_turn>"], b(0x0A)], [b(0x80)], []]
    cases = {"variants": [], "errors": [{k: e[k] for k in ("model", "normalizer", "pre_tokenizer", "decoder",
                                                          "added_tokens")} for e in errors(d)]}
    for name, v in variants(d).items():
        tok = Tokenizer.from_str(json.dumps(v))
        enc = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in TEXTS]
        for c in enc:
            c["decoded"] = tok.decode(c["ids"], skip_special_tokens=False)
        dec = [{"ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)} for ids in decode_ids]
        cases["variants"].append({"name": name, "model": v["model"] if name != "gemma" else None,
                                  "normalizer": v["normalizer"], "cases": enc, "decode": dec})
    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print({v["name"]: sum(len(c["ids"]) for c in v["cases"]) for v in cases["variants"]})


if __name__ == "__main__":
    main()
