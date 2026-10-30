"""Spanish and multilingual prompts (ingot-dev rowscal, rowscheck and Flash-Next prompts_fn.py) with non-English text.

Text of Spanish markdown pages (the files named in PAGES, read from --text-dir) trimmed to ~4K and ~16K tokens with a final
question in Spanish, a programming task explained in Spanish and a short question. Qwen3.6 chat template
with reasoning, as in prompts.py. Writes the ids to out/prompts-es/<name>.json.

Usage:
  .venv/bin/python prompts_es.py --text-dir <dir>
"""
import argparse
import json
import os

from transformers import AutoTokenizer

from moe_ref import resolve, HERE

PAGES = ["concepts/fundamentos.md", "concepts/speculative-decoding.md", "concepts/quantization.md",
         "concepts/moe-vs-dense.md", "concepts/uncensored-abliterated.md", "concepts/memory-budget.md",
         "concepts/ideas-rendimiento.md", "engines/splash.md", "engines/mlx-serve.md", "engines/llama-cpp.md",
         "models/qwen3.6-35b-a3b.md", "models/qwen3.8-27b.md", "howto/benchmarking.md", "howto/pi.md"]
TARGETS = {"es-4K": 13000, "es-16K": 50000}
QUESTION = ("\n\nCon lo que dicen estas notas, explica en español y con tus palabras qué es la decodificación "
            "especulativa, por qué la cuantización afecta a la calidad y qué modelo recomendarías para programar en "
            "este Mac. Justifica cada punto.")
CODE = ("Necesito un programa en Python que lea un fichero CSV con columnas fecha, producto y cantidad, agrupe las "
        "ventas por mes y por producto, y muestre una tabla ordenada de mayor a menor. Si una fila está mal formada, "
        "debe avisar con el número de línea y seguir. Escribe el código completo con comentarios en español y "
        "explica después, paso a paso y en español, cómo funciona y cómo lo probarías.")
CHAT = "Cuéntame en unos párrafos la historia de la imprenta en España y por qué fue importante para la lengua castellana."

MULTI = {
    "en-chat": "Tell me in a few paragraphs about the history of the printing press in England and why it mattered "
               "for the English language.",
    "code-c": "Write a complete C program that reads integers from standard input, builds a binary search tree, "
              "prints the values in order, and frees all memory. Add brief comments.",
    "ca-chat": "Explica'm en uns quants paràgrafs la història de la impremta a Catalunya i per què va ser important per "
               "a la llengua catalana.",
    "zh-chat": "请用几段话介绍一下中国印刷术的历史，以及它对汉语和中国文化的重要性。",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/Models/Qwen3.6-35B-A3B-Heretic-MLX-4bit-calib")
    ap.add_argument("--text-dir", required=True, help="folder with the Spanish markdown pages named in PAGES")
    ap.add_argument("--out", default="prompts-es")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(resolve(a.model))
    corpus = "\n\n".join(open(os.path.join(os.path.expanduser(a.text_dir), p)).read() for p in PAGES)
    out = os.path.join(HERE, "out", a.out)
    os.makedirs(out, exist_ok=True)
    prompts = {"chat": CHAT, "es-chat": CHAT, "es-code": CODE}   # chat: ingot-dev gen warm-up
    # Texts of other kinds: English, code, Catalan and Chinese (another script).
    prompts.update(MULTI)
    for name, chars in TARGETS.items():
        prompts[name] = corpus[:chars] + QUESTION
    # With reasoning the model thinks in English even if the question is in Spanish: the "-nt" variants (without
    # reasoning, enable_thinking=False) answer directly in Spanish.
    jobs = [(n, t, True) for n, t in prompts.items()] + [(n + "-nt", t, False) for n, t in prompts.items() if n != "chat"]
    for name, text, think in jobs:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                      enable_thinking=think)
        if not isinstance(ids, list):
            ids = ids["input_ids"]
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump({"name": name, "chars": len(text), "ids": list(map(int, ids))}, f)
        print(f"{name}: {len(ids)} tokens; ids ≥ 151680: {sum(i >= 151680 for i in ids)}, ≥ 100032: {sum(i >= 100032 for i in ids)}")


if __name__ == "__main__":
    main()
