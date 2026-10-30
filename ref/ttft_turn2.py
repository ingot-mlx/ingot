"""First token of the second pi turn with lab serve running (prefix cache).

Turn 1: the body of a real pi request (system prompt and 8 tools) with one user message, at
temperature 0. Turn 2: the same conversation with the reply as is and a new message of ~L tokens: a tool result
(if turn 1 called one) or a user message. Before each turn 2, turn 1 is repeated (it comes
from the snapshot, cheap) so the live state is again that of the reply. Measures on the client the time to the
first delta with text or reasoning (streaming) and prints the new tokens read (prompt - cached_tokens).

Usage:
  python3 ttft_turn2.py --body dump/0001-body.json [--url http://127.0.0.1:8090] [--lens 20,50,100,300] [--reps 3]
"""
import argparse
import json
import time
import urllib.request

FILLER = ("El fichero src/server.py define la clase Handler, que lee la petición, valida los campos obligatorios y "
          "devuelve un JSON con el resultado; los tests de tests/test_server.py cubren los casos de error. ")


def stream(url, body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, reasoning, content, calls, usage = None, "", "", {}, None
    with urllib.request.urlopen(req) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.perf_counter() - t0
                reasoning += d.get("reasoning_content") or ""
                content += d.get("content") or ""
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    c["id"] += tc.get("id") or ""
                    f = tc.get("function", {})
                    c["function"]["name"] += f.get("name") or ""
                    c["function"]["arguments"] += f.get("arguments") or ""
    return ttft, reasoning, content, [calls[k] for k in sorted(calls)], usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--body", required=True)
    ap.add_argument("--user", default=None, help="replaces the user message of turn 1")
    ap.add_argument("--lens", default="20,50,100,300")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    base = json.load(open(a.body))
    base = {k: v for k, v in base.items() if k in ("messages", "tools", "chat_template_kwargs")}
    if a.user:
        base["messages"][-1] = {"role": "user", "content": a.user}
    base.update(model="x", temperature=0, max_tokens=1024)
    _, reasoning, content, calls, u1 = stream(a.url, base)
    kind = "tool" if calls else "user"
    print(f"turn 1: {u1['prompt_tokens']} prompt tokens, {u1['completion_tokens']} generated, "
          f"{len(calls)} calls ({', '.join(c['function']['name'] for c in calls)}); turn 2 with '{kind}' message")
    asst = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        asst["tool_calls"] = calls
    for L in [int(x) for x in a.lens.split(",")]:
        text = (FILLER * (L // 40 + 2))
        text = " ".join(text.split()[: int(L * 0.72)])   # ~1.4 tokens per word
        if kind == "tool":
            new = [{"role": "tool", "tool_call_id": c["id"], "content": text} for c in calls[:1]]
        else:
            new = [{"role": "user", "content": text}]
        res = []
        for _ in range(a.reps):
            stream(a.url, base)
            body = dict(base, messages=base["messages"] + [asst] + new, max_tokens=8)
            ttft, _, _, _, u = stream(a.url, body)
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            res.append((ttft, u["prompt_tokens"] - cached, cached))
        ts = sorted(r[0] for r in res)
        print(f"L={L:3d}: {res[0][1]} new tokens (cached {res[0][2]}); first token median {ts[len(ts) // 2] * 1e3:.0f} ms "
              f"(all: {', '.join(f'{t * 1e3:.0f}' for t in ts)})")


if __name__ == "__main__":
    main()
