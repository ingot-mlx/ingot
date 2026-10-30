"""Tool-call parser references for the dialects, without MLX and without weights.

For each repo whose template writes a parsed tool format (GLM-4.7 XML, Hermes JSON, Gemma 4, Muse ATEM, LFM3's
Qwen3-Coder XML) it writes Tests/IngotServeTests/Resources/<repo name>/tools.json with:

- "spans": round trip with the template. Each assistant turn of the serve_ref.py conversations (and one with awkward
  values) is rendered after its prefix; the text after the prefix's generation prompt, up to the first end token, is
  what the model learns to write. "expected" is the message it came from (reasoning and content only when the template
  writes them) and "mlx" the calls mlx-lm's parser reads from the same text.
- "hard": hand-written outputs (unclosed call, several in a row, newlines and quotes in values, numbers where a string
  is declared, text before and after) with what mlx-lm's parser reads.

gpt-oss (harmony) has no parser in mlx-lm 0.32.2 (its server returns the raw channels): only the round trip, with
"mlx" null; its hand-written cases are in Tests/IngotServeTests/HarmonyTests.swift. Its template writes one call per
assistant turn (the first), so a turn with two only expects the first.

mlx-lm's parsers (0.32.2, from ref/.venv) are loaded by file path with importlib so that mlx_lm/__init__.py (which
imports MLX) never runs. The calls are read the way mlx_lm/server.py does: the text between the format's start and
end marks (or up to the end of the output if the call is not closed), one parse_tool_call per block; a block the
parser rejects is dropped, as the server does. Arguments a parser leaves as a JSON string are decoded.

Usage (same versions as ref/.venv, outside the GPU lock):
  uv run --no-project --with transformers==5.20.1 --with tokenizers==0.23.3 --with jinja2 --with regex \
      python -I ref/tool_ref.py
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tok_ref  # noqa: E402  (resolve, import_serve_ref, adapt)

PARSERS = os.path.join(HERE, ".venv", "lib", "python3.13", "site-packages", "mlx_lm", "tool_parsers")


def load_parser(name):
    """mlx_lm.tool_parsers.<name> without importing mlx_lm."""
    pkg = "mlx_lm_tool_parsers"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(pkg, os.path.join(PARSERS, "__init__.py"),
                                                      submodule_search_locations=[PARSERS])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
    full = f"{pkg}.{name}"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(full, os.path.join(PARSERS, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
    return sys.modules[full]


# repo -> (format as ToolCallFormat.rawValue, mlx-lm parser module)
REPOS = {
    "zai-org/GLM-4.7-Flash": ("glm47", "glm47"),
    "ibm-granite/granite-4.1-3b": ("hermes", "json_tools"),
    "google/gemma-4-26B-A4B-it": ("gemma4", "gemma4"),
    "meta-models/Muse-Glimmer-30B": ("atem", "muse_spark"),
    "openai/gpt-oss-20b": ("harmony", None),
    "LiquidAI/LFM3-1.7B": ("qwenXML", "qwen3_coder"),
}

AWKWARD = {
    "role": "assistant", "content": "",
    "reasoning_content": "Pienso \"algo\".\nDos líneas.",
    "tool_calls": [
        {"id": "call_7", "type": "function", "function": {"name": "bash", "arguments": {
            "command": "echo \"hola, {mundo}\" > f.txt\ncat f.txt", "timeout": 5}}},
        {"id": "call_8", "type": "function", "function": {"name": "read", "arguments": {
            "path": "/tmp/«x» y.txt", "offset": 1.5, "flags": ["a", "b c"]}}},
    ],
}

# Hand-written outputs per format (after the generation prompt), for the comparison with mlx-lm.
HARD = {
    "glm47": [
        ("text-around", 'Texto antes.<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la\necho "x"</arg_value>'
                        "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>Texto después."),
        ("two-calls", "<tool_call>read<arg_key>path</arg_key><arg_value>/a</arg_value></tool_call>"
                      '<tool_call>read<arg_key>path</arg_key><arg_value>/b</arg_value><arg_key>flags</arg_key>'
                      '<arg_value>["x", "y"]</arg_value></tool_call>'),
        ("unclosed", "Miro.<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value>"),
        ("number-as-string", "<tool_call>bash<arg_key>command</arg_key><arg_value>42</arg_value>"
                             "<arg_key>timeout</arg_key><arg_value>7</arg_value></tool_call>"),
        ("newlines-between", "<tool_call>read\n<arg_key>path</arg_key>\n<arg_value>/x</arg_value>\n"
                             "<arg_key>offset</arg_key>\n<arg_value>2.5</arg_value>\n</tool_call>"),
        ("reasoning-then-call", "Razono.</think><tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value>"
                                "</tool_call>"),
        # A file ending in a newline, as the model writes it: the template writes the value back verbatim.
        ("trailing-newline", "<tool_call>bash<arg_key>command</arg_key><arg_value>cat > f.py <<EOF\nprint(1)\nEOF\n"
                             "</arg_value><arg_key>timeout</arg_key><arg_value>\n7\n</arg_value></tool_call>"),
    ],
    "hermes": [
        ("text-around", 'Antes.\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la\\necho \\"x\\"", '
                        '"timeout": 30}}\n</tool_call>\nDespués.'),
        ("two-calls", '<tool_call>\n{"name": "read", "arguments": {"path": "/a"}}\n</tool_call>\n'
                      '<tool_call>\n{"name": "read", "arguments": {"path": "/b", "flags": ["x"]}}\n</tool_call>'),
        ("unclosed", '<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}'),
        ("arguments-string", '<tool_call>\n{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}\n</tool_call>'),
        ("number-as-string", '<tool_call>\n{"name": "bash", "arguments": {"command": 42, "timeout": "7"}}\n'
                             "</tool_call>"),
        ("reasoning-opened", "<think>\nPienso.\n</think>\n\n<tool_call>\n"
                             '{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'),
    ],
    "gemma4": [
        ("text-around", 'Antes.<|tool_call>call:read{flags:[<|"|>a<|"|>,<|"|>b, c<|"|>],path:<|"|>/t{x}.txt<|"|>}'
                        "<tool_call|>Después."),
        ("two-calls", '<|tool_call>call:read{path:<|"|>/a<|"|>}<tool_call|><|tool_call>call:bash{command:<|"|>ls<|"|>,'
                      "timeout:3}<tool_call|>"),
        ("unclosed", '<|tool_call>call:bash{command:<|"|>pwd<|"|>}'),
        ("quotes-newlines", '<|tool_call>call:bash{command:<|"|>echo "hi"\nls<|"|>,timeout:5}<tool_call|>'),
        ("number-as-string", "<|tool_call>call:bash{command:42,timeout:7}<tool_call|>"),
        ("nested", '<|tool_call>call:read{flags:[],offset:0.5,path:<|"|>x<|"|>}<tool_call|>'),
        ("trailing-newline", '<|channel>thought\nPienso.\n<channel|><|tool_call>call:bash{command:<|"|>cat > f.py'
                             ' <<EOF\nprint(1)\nEOF\n<|"|>}<tool_call|>'),
    ],
    # Qwen3-Coder XML as LFM3's template writes it: reasoning closed with no newline, calls right after the
    # content, several separated by "\n\n", each value between newlines.
    "qwenXML": [
        ("text-around", "Antes.<tool_call>\n<function=bash>\n<parameter=command>\nls -la\necho \"x\"\n</parameter>\n"
                        "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"),
        ("two-calls", "<tool_call>\n<function=read>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>\n\n"
                      "<tool_call>\n<function=read>\n<parameter=path>\n/b\n</parameter>\n<parameter=flags>\n"
                      "[\"x\", \"y\"]\n</parameter>\n</function>\n</tool_call>"),
        ("number-as-string", "<tool_call>\n<function=bash>\n<parameter=command>\n42\n</parameter>\n"
                             "<parameter=timeout>\n7\n</parameter>\n</function>\n</tool_call>"),
        ("float-and-list", "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/«x» y.txt\n</parameter>\n"
                           "<parameter=offset>\n1.5\n</parameter>\n<parameter=flags>\n[\"a\", \"b c\"]\n</parameter>\n"
                           "</function>\n</tool_call>"),
        ("reasoning-then-call", "Razono.</think><tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                                "</function>\n</tool_call>"),
        # A file ending in a newline: the template writes "\n" + value + "\n", so the value keeps its own newline.
        ("trailing-newline", "<tool_call>\n<function=bash>\n<parameter=command>\ncat > f.py <<EOF\nprint(1)\nEOF\n\n"
                             "</parameter>\n</function>\n</tool_call>"),
    ],
    "atem": [
        ("reasoning-call", ' to=self<|message|>Pienso.<|eom|><|start|>assistant to=bash<|message|>'
                           '<atem:function_calls>\n<atem:invoke name="bash">\n<atem:parameter name="command">ls -la'
                           "</atem:parameter>\n<atem:parameter name=\"timeout\">30</atem:parameter>\n</atem:invoke>\n"
                           "</atem:function_calls>"),
        ("two-invokes", ' to=read<|message|><atem:function_calls>\n<atem:invoke name="read">\n'
                        '<atem:parameter name="path">/a</atem:parameter>\n</atem:invoke>\n<atem:invoke name="read">\n'
                        '<atem:parameter name="path">/b</atem:parameter>\n<atem:parameter name="flags">["x"]'
                        "</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"),
        ("unclosed", ' to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">\n'
                     '<atem:parameter name="command">pwd</atem:parameter>\n'),
        ("number-as-string", ' to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">\n'
                             '<atem:parameter name="command">42</atem:parameter>\n</atem:invoke>\n'
                             "</atem:function_calls>"),
        ("spaces-kept", ' to=bash<|message|><atem:function_calls>\n<atem:invoke name="bash">\n'
                        '<atem:parameter name="command">  echo "a"\n</atem:parameter>\n</atem:invoke>\n'
                        "</atem:function_calls>"),
        ("answer", " to=self<|message|>Breve.<|eom|><|start|>assistant to=user<|message|>Hola, ¿qué tal?"),
    ],
}


def marks(fmt, mod):
    return mod.tool_call_start, mod.tool_call_end


def mlx_calls(text, fmt, mod, tools):
    """Calls mlx_lm/server.py would return for this output (tool blocks only; None without a parser)."""
    if mod is None:
        return None
    start, end = marks(fmt, mod)
    if fmt != "atem" and "</think>" in text:
        text = text.split("</think>", 1)[1]
    if fmt == "gemma4" and "<channel|>" in text:
        text = text.split("<channel|>", 1)[1]
    calls, rest = [], text
    while start in rest:
        rest = rest.split(start, 1)[1]
        block, sep, rest = rest.partition(end)
        try:
            parsed = mod.parse_tool_call(block, tools)
        except Exception:
            parsed = None
        if parsed is not None:
            for c in parsed if isinstance(parsed, list) else [parsed]:
                args = c.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                calls.append({"name": c.get("name"), "arguments": args})
        if not sep:
            break
    return calls


def end_strings(tok, repo):
    return [s for s in tok.convert_ids_to_tokens(tok_ref.end_tokens(repo, tok)) if s]


def spans(repo, fmt, mod, tok, serve_ref):
    template = tok.chat_template if isinstance(tok.chat_template, str) else ""
    multi = serve_ref.MULTI
    turns = [("turn1", multi[:2], multi[2]), ("turn2", multi[:4], multi[4]), ("turn3", multi[:7], multi[7]),
             ("awkward", multi[:2], AWKWARD)]
    if fmt == "harmony":
        # turn3 drops the earlier analysis from the history (not a continuation): the answer straight after the user.
        turns.append(("answer", multi[:2], multi[7]))
    ends = end_strings(tok, repo)
    out, skipped = [], []
    for kw0, suffix in [({}, ""), ({"enable_thinking": True}, "-think"), ({"enable_thinking": False}, "-nothink")]:
        for name, prefix, msg in turns:
            m = dict(msg)
            if suffix == "-nothink":
                m.pop("reasoning_content", None)
            msgs, kw = tok_ref.adapt(prefix + [m], kw0, template)
            try:
                gen = tok.apply_chat_template(msgs[:-1], tools=serve_ref.TOOLS, add_generation_prompt=True,
                                              tokenize=False, **kw)
                full = tok.apply_chat_template(msgs, tools=serve_ref.TOOLS, add_generation_prompt=False,
                                               tokenize=False, **kw)
            except Exception as e:
                skipped.append(f"{name}{suffix}: {str(e)[:80]}")
                continue
            start = len(gen)
            if not full.startswith(gen):
                # Thinking off: the generation prompt may close an empty reasoning block that the turn, rendered as
                # history, does not write (Gemma 4). The model writes what follows the common prefix.
                plain = tok.apply_chat_template(msgs[:-1], tools=serve_ref.TOOLS, add_generation_prompt=False,
                                                tokenize=False, **kw)
                start = len(os.path.commonprefix([gen, full]))
                tag = full.rfind("<", 0, start)  # do not split a tag both texts start alike
                if tag >= 0 and ">" not in full[tag:start]:
                    start = tag
                if suffix != "-nothink" or not full.startswith(plain) or start < len(plain):
                    skipped.append(f"{name}{suffix}: the turn does not continue the generation prompt")
                    continue
            text = full[start:]
            cut = min([text.find(e) for e in ends if e in text] or [len(text)])
            text = text[:cut].rstrip()
            if any(o["text"] == text and o["prompt_tail"] == gen[-64:] for o in out):
                continue
            reasoning = (msg.get("reasoning_content") or "").strip() if suffix != "-nothink" else ""
            calls = msg.get("tool_calls", [])[:1 if fmt == "harmony" else None]
            content = (msg.get("content") or "").strip()
            if fmt == "harmony" and calls and not reasoning:
                reasoning, content = content, ""  # the template writes a call's content as its analysis
            expected = {
                "reasoning": reasoning if reasoning and reasoning in text else "",
                "content": content if content and content in text else "",
                "calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                          for c in calls],
            }
            out.append({"name": name + suffix, "prompt_tail": gen[-64:], "text": text, "expected": expected,
                        "mlx": mlx_calls(text, fmt, mod, serve_ref.TOOLS)})
    return out, skipped


def main():
    from transformers import AutoTokenizer
    tok_ref.pin_clock()  # gpt-oss writes today's date
    serve_ref = tok_ref.import_serve_ref()
    for repo, (fmt, parser) in REPOS.items():
        try:
            path = tok_ref.resolve(repo)
        except OSError:
            print(f"{repo}: not in the Hugging Face cache, skipped")
            continue
        mod = load_parser(parser) if parser else None
        tok = AutoTokenizer.from_pretrained(path)
        sp, skipped = spans(repo, fmt, mod, tok, serve_ref)
        hard = [{"name": n, "text": t, "mlx": mlx_calls(t, fmt, mod, serve_ref.TOOLS)} for n, t in HARD.get(fmt, [])]
        out = os.path.join(HERE, "..", "Tests", "IngotServeTests", "Resources", repo.split("/")[-1])
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "tools.json"), "w") as f:
            json.dump({"format": fmt, "tools": serve_ref.TOOLS, "spans": sp, "hard": hard}, f, ensure_ascii=False,
                      indent=1)
        agree = "no mlx-lm parser" if mod is None else \
            f"mlx-lm agrees with the template on {sum(s['mlx'] == s['expected']['calls'] for s in sp)}/{len(sp)}"
        print(f"{repo} ({fmt}): {len(sp)} spans, {len(hard)} hard cases; {agree}")
        for s in skipped:
            print(f"  skipped {s}")


if __name__ == "__main__":
    main()
