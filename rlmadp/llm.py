"""The two model seams, real and offline.

`RLM` takes a `root` and a `sub` as plain callables, so this module supplies
either a real OpenAI-compatible server (vLLM, Ollama, anything) or an offline
stand-in for tracing the loop without a GPU.

Be clear about what the offline pair is: it is NOT a language model, and it is
not evidence that anything works. `scripted_root` replays a fixed transcript,
and `extractive_sub` is a keyword-overlap sentence picker. They exist so you can
watch the machinery -- the REPL, the caching, the coverage accounting, the
grounding guard -- run end to end in half a second. Judge behaviour only with
`--vllm`.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from typing import Callable, Optional

# ---------------------------------------------------------------- real models


# A server that silently drops most of a slice is the worst failure this project
# has: the root locates the right region, the sub answers NOT FOUND, and nothing
# reports an error. Ollama ships a 2048-token default window and truncates from
# the FRONT, so a 40,000-char slice arrives as its last ~8,000 chars. English
# prose runs ~4 chars/token and even dense text stays under ~6, so a server
# reporting far more than that saw far less text than we sent.
TRUNCATION_CHARS_PER_TOKEN = 8.0


def openai_compatible(
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    api_key: str = "EMPTY",
    max_tokens: int = 3072,
    timeout: int = 180,
    sub_model: Optional[str] = None,
) -> tuple[Callable, Callable]:
    """Return (root_fn, sub_fn) speaking to any OpenAI-compatible endpoint.

    `sub_model` defaults to `model`, but the two roles want different things.
    The ROOT writes Python and follows a multi-step protocol, so it needs
    instruction-following and code ability. The SUB only reads one slice and
    answers one question, and it is called with a very large prompt -- so a
    smaller, faster model there is usually the right trade.
    """
    sub_name = sub_model or model
    warned = {"done": False}

    def _chat(messages: list[dict], n_tokens: int, name: str) -> str:
        body = json.dumps(
            {"model": name, "messages": messages, "max_tokens": n_tokens, "temperature": 0.0}
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())

        # Compare what we sent against what the server says it saw.
        sent = sum(len(m["content"]) for m in messages)
        seen = (payload.get("usage") or {}).get("prompt_tokens")
        if seen and not warned["done"] and sent / seen > TRUNCATION_CHARS_PER_TOKEN:
            warned["done"] = True
            print(
                f"\n  !! SERVER TRUNCATED THE PROMPT: sent {sent:,} chars, server "
                f"counted {seen:,} tokens ({sent / seen:.1f} chars/token -- prose is ~4).\n"
                f"     Its context window is too small and it dropped the FRONT of the\n"
                f"     slice. Results below are meaningless. For ollama, restart it as:\n"
                f"       OLLAMA_CONTEXT_LENGTH=32768 ollama serve\n"
                f"     or bake it in:  ollama create {name}-32k -f <Modelfile with\n"
                f"       PARAMETER num_ctx 32768>\n",
                file=sys.stderr,
            )
        return payload["choices"][0]["message"]["content"]

    def root(messages: list[dict]) -> str:
        # The root writes CODE, and a verbose model writes a lot of it. At 1024
        # tokens a Qwen3-4B first turn was cut off mid-expression, so the block
        # never got its closing fence, CODE_BLOCK matched nothing, and the turn
        # was spent on a "no code block found" nudge instead of on the document.
        # Truncation here is invisible in the transcript -- the reply just looks
        # like the model forgot to close the fence.
        return _chat(messages, max_tokens, model)

    def sub(question: str, text: Optional[str]) -> str:
        # The sub gets a FRESH message list every time. No history, no
        # accumulation -- this is the clean window that makes recursion work.
        prompt = question if text is None else f"{question}\n\n---\n{text}"
        return _chat(
            [
                {
                    "role": "system",
                    "content": "Answer using ONLY the text provided. Be concise and factual. "
                    "If the text does not contain the answer, say exactly: NOT FOUND.",
                },
                {"role": "user", "content": prompt},
            ],
            192,  # sub answers must be SHORT -- they go back into the root's context
            sub_name,
        )

    return root, sub


# -------------------------------------------------------------- offline pair

_STOP = {
    "what", "which", "who", "when", "where", "the", "and", "was", "were", "did", "does",
    "with", "that", "this", "from", "his", "her", "for", "after", "before", "name", "year",
    "give", "answer", "text", "based", "note", "truncated", "chars", "total",
}


def extractive_sub() -> Callable[[str, Optional[str]], str]:
    """A keyword-overlap sentence picker standing in for a sub-model.

    Real enough to demonstrate the ONE property that matters here: it can only
    see the slice it is handed. Hand it the wrong region and it returns nothing,
    exactly as a real sub-model would.
    """

    def sub(question: str, text: Optional[str]) -> str:
        if not text:
            return "NOT FOUND (no text supplied)"
        keys = {w for w in re.findall(r"[a-z']+", question.lower()) if len(w) > 3 and w not in _STOP}
        sents = re.split(r"(?<=[.!?])\s+", text)
        best: list[tuple[int, int]] = []
        for n, sent in enumerate(sents):
            # Word boundaries, not substring: 'rana' otherwise matches every
            # 'Maharana' in the filler, and the picker happily returns noise.
            low = sent.lower()
            score = sum(1 for k in keys if re.search(rf"\b{re.escape(k)}\b", low))
            if score:
                best.append((score, n))
        if not best:
            return "NOT FOUND"
        best.sort(key=lambda p: -p[0])
        # Return the best sentence AND the one after it. A real sub-model
        # summarising a passage carries the follow-on clause; picking a lone
        # sentence drops exactly the kind of trailing detail ("...and died on
        # the nineteenth of January 1597") that the next hop needs.
        n = best[0][1]
        return " ".join(s.strip() for s in sents[n:n + 2])[:600]

    return sub


# A canned session. Each entry is one assistant reply, exactly as a root model
# would write it. Read them top to bottom -- they ARE the walkthrough.
_SCRIPT = [
    # 1. Cheap orientation. Never read the document; measure it.
    """I'll size the document and look at its head.
```python
print(len(context))
print(context[:180])
```""",
    # 2. Distinctive terms, each searched SEPARATELY. Note "Bhamashah" fails.
    """Now I'll probe for distinctive terms. Searching the question itself would be useless.
```python
for t in ["Haldighati", "Bhamashah", "minister", "treasury"]:
    hits = [m.start() for m in re.finditer(t.lower(), context.lower())]
    print(f"{t:12s} {len(hits):4d} {hits[:4]}")
```""",
    # 3. Zero hits means the PATTERN is wrong. Loosen it, then read big.
    """"Bhamashah" got zero hits -- likely a spelling variant, not an absent fact.
Trying a loose pattern, then reading the whole region around it.
```python
m = re.search(r"bhama\\s*shah", context, re.I)
i = m.start()
print("found at", i)
ans1 = llm_query(
    "Which minister restored Maharana Pratap's finances after Haldighati, "
    "and what did his donation pay for?",
    context[max(0, i - 20000): i + 20000])
print(ans1)
```""",
    # 4. HOP: the name that came BACK from the sub becomes the next search key.
    """Hop 1 gives me "Bhama Shah" -- a term I did not have before. Now I can
search for it, which I could not do one turn ago.
```python
hits = [m.start() for m in re.finditer(r"bhama\\s*shah", context, re.I)]
print("Bhama Shah occurrences:", hits)
j = hits[1]
ans2 = llm_query(
    "What seat or capital did Pratap Singh fix after recovering Mewar?",
    context[max(0, j - 20000): j + 20000])
print(ans2)
```""",
    # 5. HOP again: "Chavand" was unknowable until hop 2 returned it.
    """Hop 2 gives me "Chavand". Searching for it now.
```python
hits = [m.start() for m in re.finditer("chavand", context, re.I)]
print("Chavand occurrences:", hits)
k = hits[-1]
ans3 = llm_query(
    "In what year did the Rana die at Chavand?",
    context[max(0, k - 20000): k + 20000])
print(ans3)
```""",
    # 6. VERIFY the hops resolved, then commit. The prompt asks for this
    #    explicitly, and it is what separates "I read it" from "I assumed it".
    """Before committing I'll check both hops actually resolved from what I read.
```python
capital = "Chavand" if "Chavand" in ans2 else None
year = re.search(r"\\b1[56]\\d\\d\\b", ans3)
print("hop2 capital:", capital, "| hop3 year:", year.group() if year else None)
if capital and year:
    FINAL(f"Capital: {capital}; Pratap died there in {year.group()}")
else:
    FINAL_NONE("the slices I read did not contain both hops")
```""",
]


def scripted_root() -> Callable[[list[dict]], str]:
    """Replay `_SCRIPT`, ignoring the conversation. Not a model."""
    turns = iter(_SCRIPT)

    def root(messages: list[dict]) -> str:
        return next(turns, "```python\nFINAL_NONE('script exhausted')\n```")

    return root
