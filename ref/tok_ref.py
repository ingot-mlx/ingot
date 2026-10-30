"""Tokenizer and chat template references for the model families, without MLX and without weights:
Hugging Face tokenizers ids and decode for each repo's tokenizer.json, and transformers' apply_chat_template, read from
the Hugging Face cache. Writes Tests/IngotServeTests/Resources/<repo name>/{tok,chat,special,model}.json.

chat.json and special.json are those of serve_ref.py (same conversations, special.json with the template-only ids),
adapted to each template: gpt-oss reads the reasoning from `thinking` instead of `reasoning_content`, Muse gets a
fixed `current_date`, and a conversation the template rejects is left out (printed). gpt-oss writes today's date
with strftime_now: transformers' clock is pinned to FIXED_NOW (local time), the same instant
TokenizerTemplateTests gives ChatTemplate.now, so the references do not depend on the day they are generated or
checked. model.json has the end tokens (generation_config, config and the tokenizer's eos_token).

The corpus is the one of serve_ref.py (its 22 texts and 30,000 characters of asyncio sources) plus cases aimed at the
new tokenizers: upper-case contractions, combining marks, CRLF, runs of U+2581 and newlines, characters outside the
Gemma vocabulary (they go to <0xNN>) and a long text without specials. serve_ref.py imports moe_ref, which imports
MLX: its data lists are reused here with moe_ref stubbed out.

Usage (same versions as ref/.venv, outside the GPU lock):
  uv run --no-project --with tokenizers==0.23.3 --with transformers==5.20.1 --with jinja2 --with regex \
      python -I ref/tok_ref.py [--repo org/name ...]
"""
import argparse
import datetime
import glob
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))

# Tokenizer, template and config files of these repos (no weights) live in the Hugging Face cache for the CI: one repo
# per tier 1 family (wiki plan-motor, "Que familias entran"). froggeric's repo has the Heretic's tokenizer and template
# (the calibrated model took them from it, byte for byte); Flash-Next's template is the Qwen3.8 27B's.
REPOS = [
    "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit",
    "Qwen/Qwen3.8-Flash-Next",
    "LiquidAI/LFM2.5-8B-A1B-MLX-4bit",
    "google/gemma-4-26B-A4B-it",
    "meta-models/Muse-Glimmer-30B",
    "openai/gpt-oss-20b",
    "zai-org/GLM-4.7-Flash",
    "ibm-granite/granite-4.1-3b",
    "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit",  # tokenizer.json and template identical to nvidia's BF16 repo
    # LFM3: same template and tokens as the CI's LFM3-1.7B-MLX-8bit (its tokenizer.json writes the merges
    # as lists; the MLX one as strings).
    "LiquidAI/LFM3-1.7B",
]

EXTRA_TEXTS = [
    "DON'T WE'LL THEY'RE I'M SHE'D YOU'VE IT'S. Don'T We'Ll",
    "Marcas: e\u0301 a\u0308 n\u0303 o\u0302\u0323 क\u093fन\u094d ש\u05c1\u05b8 Z\u0351\u036b\u0343\u036a",
    "line one\r\nline two\r\n\r\n  \r\n\tend\r",
    "\u2581\u2581\u2581 hola \u2581\u2581 mundo\u2581",
    "\n" * 50 + "x" + " " * 40 + "y" + "\t" * 9 + "\n\n \n",
    "Fuera de vocabulario: \ue000\uf8ff \U00020000\U0002a6a5\U00030b5e \x00\x01\x1f\x7f \U0010fffd \U0001fae8\U0001fa77 \U00040000\U00050001 (plane 4-5: byte f1, missing in Granite)",
    "<table><tr><td>celda</td></tr></table>\n<div class=\"x\">  <span>texto</span></div>\n\n\n\n",
    "    indented code\n        deeper\n            and deeper\n",
]


# Texts aimed at one repo's pre-tokenizer, added after the common ones (only that repo's tok.json changes).
REPO_TEXTS = {
    # Nemotron-H: its Split regex isolates every digit (\p{N}), splits on the letter classes Lu/Lt/Lm/Lo/M
    # against Ll/Lm/Lo/M, glues [\r\n/]* after punctuation runs, and the BPE has ignore_merges (whole words that are in
    # the vocabulary are taken as one token before merging).
    "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit": [
        "1234567890 3.14159 2026-10-26 v0.32.4 1,000,000 0x7fff ½ ² ³ Ⅻ ٣٤٥ ０１２",
        "Title case: ǅungla ǈubav ǋaš ᾈᾉ; modifiers: ʰʲˀ ʰa aʰ; Lo: 日本語テキスト 中文 العربية עברית ไทย",
        "camelCaseWords HTTPServer XMLHttpRequest iPhone McDonald's ÉCOLE école ÜberÄrger",
        "a/b/c ..//\n// end;\r\n/* comment */\n\n/// doc\r\n--/\n?!/\r",
        " the the  the   Hello hello HELLO internationalization antidisestablishmentarianism",
    ],
}


def import_serve_ref():
    """serve_ref.py with a stub moe_ref (only resolve and HERE are used): no MLX import."""
    stub = types.ModuleType("moe_ref")
    stub.resolve, stub.HERE = resolve, HERE
    sys.modules["moe_ref"] = stub
    sys.path.insert(0, HERE)
    import serve_ref
    return serve_ref


def resolve(repo):
    repo = os.path.expanduser(repo)
    if os.path.isdir(repo):
        return repo
    base = os.path.join(HUB, "models--" + repo.replace("/", "--"))
    ref = os.path.join(base, "refs", "main")
    if os.path.exists(ref):
        return os.path.join(base, "snapshots", open(ref).read().strip())
    # As ModelPaths.resolve: without refs/main (a snapshot copied by hand), the first snapshot.
    return os.path.join(base, "snapshots", sorted(os.listdir(os.path.join(base, "snapshots")))[0])


def texts(serve_ref):
    corpus = "".join(open(f).read() for f in sorted(glob.glob(os.path.expanduser(
        "~/.local/share/mise/installs/python/3.14/lib/python3.14/asyncio") + "/*.py")))
    return list(serve_ref.TEXTS) + [corpus[:30000]] + EXTRA_TEXTS


def tok_cases(repo, all_texts):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(resolve(repo), "tokenizer.json"))
    out = []
    for t in all_texts:
        ids = tok.encode(t, add_special_tokens=False).ids
        out.append({"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    return out


MUSE_DATE = "2026-10-24"
# strftime_now's clock (naive local time, as datetime.now()); keep in sync with TokenizerTemplateTests.fixedNow.
# Noon, so that the date is the same in every time zone within ±12 h; day and month below 10 to check zero padding.
FIXED_NOW = datetime.datetime(2026, 1, 5, 12, 0, 0)


def pin_clock():
    """transformers' strftime_now calls chat_template_utils.datetime.now() when rendering: pinned to FIXED_NOW."""
    import transformers.utils.chat_template_utils as ctu

    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXED_NOW

    ctu.datetime = FixedDatetime
    assert ctu.datetime.now() == FIXED_NOW


def adapt(msgs, kwargs, template):
    """The serve_ref.py conversation as this template expects it."""
    msgs = json.loads(json.dumps(msgs))
    kwargs = dict(kwargs)
    if "message.thinking" in template:
        for m in msgs:
            if "reasoning_content" in m:
                m["thinking"] = m.pop("reasoning_content")
                if m.get("tool_calls"):
                    m["content"] = ""  # gpt-oss: content and thinking together with tool calls is an error
    if "current_date" in template:
        kwargs["current_date"] = MUSE_DATE
    return msgs, kwargs


def chat_cases(repo, serve_ref):
    from transformers import AutoTokenizer
    pin_clock()
    tok = AutoTokenizer.from_pretrained(resolve(repo))
    template = tok.chat_template if isinstance(tok.chat_template, str) else ""
    extra = [("reasoning-effort-high", [{"role": "user", "content": "Hola"}], None, {"reasoning_effort": "high"})]
    chats, specials, skipped = [], [], []
    for name, msgs0, tools, kw0 in list(serve_ref.CHATS) + extra:
        msgs, kw = adapt(msgs0, kw0, template)
        agp = kw.pop("add_generation_prompt", True)
        try:
            text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=agp, tokenize=False, **kw)
        except Exception as e:
            skipped.append(f"{name}: {str(e)[:80]}")
            continue
        if "{'type': 'text'" in text:  # the template prints the parts list; the server joins parts (normalize)
            skipped.append(f"{name}: the template does not take content parts")
            continue
        c = {"name": name, "messages": msgs, "tools": tools, "kwargs": kw, "add_generation_prompt": agp,
             "text": text, "ids": tok.encode(text, add_special_tokens=False)}
        chats.append(c)
    for name, msgs0, tools, kw0 in serve_ref.SPECIAL_CHATS:
        msgs, kw = adapt(msgs0, kw0, template)
        try:
            text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **kw)
        except Exception as e:
            skipped.append(f"special {name}: {str(e)[:80]}")
            continue
        contents = [c for c in serve_ref.strings([msgs, tools]) if len(c) > 2]
        c = {"name": name, "messages": msgs, "tools": tools, "kwargs": kw, "add_generation_prompt": True,
             "text": text, "ids": serve_ref.template_only_ids(tok, text, contents),
             "ids_parse_all": tok.encode(text, add_special_tokens=False)}
        specials.append(c)
    return chats, specials, skipped, tok


def end_tokens(repo, tok):
    eos = set()
    for f in ["generation_config.json", "config.json"]:
        path = os.path.join(resolve(repo), f)
        if not os.path.exists(path):
            continue
        cfg = json.load(open(path))
        for v in [cfg.get("eos_token_id"), cfg.get("text_config", {}).get("eos_token_id")]:
            eos.update(v if isinstance(v, list) else [] if v is None else [v])
    if tok.eos_token is not None:
        eos.add(tok.convert_tokens_to_ids(tok.eos_token))
    return sorted(eos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", help="repo id or directory (default: all of REPOS)")
    a = ap.parse_args()
    serve_ref = import_serve_ref()
    all_texts = texts(serve_ref)
    for repo in a.repo or REPOS:
        out = os.path.join(HERE, "..", "Tests", "IngotServeTests", "Resources", repo.rstrip("/").split("/")[-1])
        os.makedirs(out, exist_ok=True)
        toks = tok_cases(repo, all_texts + REPO_TEXTS.get(repo, []))
        with open(os.path.join(out, "tok.json"), "w") as f:
            json.dump(toks, f, ensure_ascii=False)
        chats, specials, skipped, tok = chat_cases(repo, serve_ref)
        for name, data in [("chat.json", chats), ("special.json", specials)]:
            with open(os.path.join(out, name), "w") as f:
                json.dump(data, f, ensure_ascii=False)
        with open(os.path.join(out, "model.json"), "w") as f:
            json.dump({"eos": end_tokens(repo, tok)}, f)
        print(f"{repo}: {len(toks)} texts, {sum(len(c['ids']) for c in toks)} ids; {len(chats)} conversations, "
              f"{len(specials)} with specials; eos {end_tokens(repo, tok)}")
        for sk in skipped:
            print(f"  skipped {sk}")


if __name__ == "__main__":
    main()
