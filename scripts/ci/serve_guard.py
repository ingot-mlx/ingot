#!/usr/bin/env python3
"""CI (--toy): one toy through lab serve's ChatServer with the agent guards on.

scripts/ci/serve_guard.py LAB TOY LOG

Starts `LAB serve --model TOY --agent-guards on` on a free port and, with a repetition penalty below 1 (it rewards
repeats, so the toy loops on purpose):
  1. a greedy request with up to 512 tokens has to stop early with finish_reason "length" and the server log has to
     carry its "guard: token loop" line;
  2. the same request with max_tokens below the cut gives the same ids, a prefix of the first (the guard only stops,
     it never changes tokens);
  3. the same request streamed ends with finish_reason "length" and the same text.
Exits 1 with the reason if anything differs. The tokens before the cut come from the toy (golden-protected); if a
deliberate token change stops the toy looping on all the prompts, pick another prompt here.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request

PROMPTS = ["Write a haiku about the sea.", "Hola", "List the files in the current directory."]
MAX_TOKENS = 512


def fail(msg):
    print(f"serve-guard: FAILED: {msg}")
    sys.exit(1)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    lab, toy, log_path = sys.argv[1:4]
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    log = open(log_path, "w")
    srv = subprocess.Popen(
        [lab, "serve", "--model", toy, "--port", str(port), "--agent-guards", "on", "--prefix-disk", "0",
         "--expert-preload", "0"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            if srv.poll() is not None:
                fail(f"lab serve exited with {srv.returncode} before answering /health (see {log_path})")
            try:
                urllib.request.urlopen(url + "/health", timeout=1).read()
                break
            except OSError:
                time.sleep(0.05)
        else:
            fail(f"lab serve did not answer /health in 300 polls (see {log_path})")
        ready = time.time() - t0

        def post(body):
            req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode()

        for prompt in PROMPTS:
            body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "repetition_penalty": 0.3,
                    "max_tokens": MAX_TOKENS}
            a = json.loads(post({**body, "lab_return_tokens": True}))
            ids = a["lab_tokens"]
            if a["choices"][0]["finish_reason"] == "length" and len(ids) < MAX_TOKENS:
                break
        else:
            fail(f"the toy did not loop on any prompt within {MAX_TOKENS} tokens: no guard trip to check")
        cut = len(ids)
        text = a["choices"][0]["message"].get("content") or ""
        b = json.loads(post({**body, "max_tokens": cut - 16, "lab_return_tokens": True}))
        if b["lab_tokens"] != ids[: cut - 16]:
            fail(f"with max_tokens {cut - 16} the ids are not a prefix of the {cut} the guard cut")
        events = [ln[6:] for ln in post({**body, "stream": True}).splitlines() if ln.startswith("data: {")]
        chunks = [json.loads(e) for e in events]
        finishes = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
        streamed = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c.get("choices"))
        if finishes[-1] != "length" or streamed != text:
            fail(f"stream: finish {finishes[-1]!r} (expected 'length'), same text {streamed == text}")
    finally:
        # The stream ends before the server logs its guard line, so wait for that line (up to 5 s) before stopping it.
        deadline = time.time() + 5
        while time.time() < deadline and sum("guard: token loop" in ln for ln in open(log_path)) < 2:
            time.sleep(0.05)
        srv.send_signal(2)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.terminate()
            srv.wait(timeout=30)
        log.close()
    trips = [ln for ln in open(log_path) if "guard: token loop" in ln]
    if len(trips) != 2:
        fail(f"{len(trips)} 'guard: token loop' lines in {log_path} (expected 2: the request and its stream)")
    print(f"serve-guard: OK ({prompt!r} cut at {cut} of {MAX_TOKENS} tokens with finish 'length', ids a prefix, "
          f"stream the same; server ready in {ready:.1f} s, {time.time() - t0:.1f} s in all)")


if __name__ == "__main__":
    main()
