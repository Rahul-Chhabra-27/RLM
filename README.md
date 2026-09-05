# rlmadp — a minimal Recursive Language Model

A standalone, dependency-free RLM in ~350 lines, written to be read rather than
run in production. No KVPress, no compression, no benchmark harness.

## The idea

Long-context models don't fail because the prompt exceeds the window. They fail
*inside* it — accuracy decays well before the hard limit as attention spreads
thin and distractors pile up. Buying a bigger window moves the cliff; it doesn't
remove it.

An RLM stops feeding the model the document at all:

> Bind the document to a Python variable. Give the model a REPL.
> Let it write code to find what it needs, and spawn **fresh** sub-models on the
> slices it selects.

The root model holds only its own short transcript. Document length never enters
its context, so degradation is decoupled from document length.

## Files

| File | What it is |
|---|---|
| `rlmadp/rlm.py` | **The whole implementation.** Read this one. |
| `rlmadp/corpus.py` | A synthetic Rajputana-history haystack with 3 facts planted far apart |
| `rlmadp/llm.py` | Model seams: a real OpenAI-compatible client, plus an offline stand-in |
| `rlmadp/cli.py` | CLI demo |
| `rlmadp/tracing.py` | Logs every trigger point in `rlm.py`, in firing order |
| `rlmadp/deepdive.py` | Shows the data crossing each boundary: REPL creation, sub-call spawn |
| `rlmadp/guards_demo.py` | Shows the two guards *rejecting* bad endings |
| `setup.sh`, `run_info.sh` | Infolab-cluster bootstrap and runner — see `INFOLAB.md` |
| `notebooks/rlm_rajputana.py` | marimo notebook — runs on [molab](https://molab.marimo.io) with no setup |

Read `rlm.py` in this order: `ROOT_SYSTEM_PROMPT` → `RLM._build_env` → `RLM.run`.

## Run it

No install needed — the core is stdlib-only.

```bash
cd ~/Desktop/rlmadp

python3 -m rlmadp.cli               # offline trace, no GPU, ~1 second
python3 -m rlmadp.tracing           # which function fires when
python3 -m rlmadp.deepdive          # what data crosses each boundary
python3 -m rlmadp.guards_demo       # watch the guards reject bad answers
python3 -m rlmadp.cli --chars 3000  # starve the slices; it abstains honestly

# against a real model
pip install -e ".[serve]"
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
python3 -m rlmadp.cli --vllm
```

On the IITB CSE infolab cluster, read **[`INFOLAB.md`](INFOLAB.md)** instead —
`$HOME` is a tight NFS quota there and envs must go on host-local disk.

**Offline mode is machinery only.** `scripted_root` replays a fixed transcript
and `extractive_sub` is a keyword sentence-picker — neither is a language model,
and neither is evidence that anything works. They exist so you can watch the
REPL, the caching, the coverage accounting and the guards run end to end without
a GPU. Judge behaviour only with `--vllm`.

## The task

The corpus plants three facts at 15%, 45% and 85% — far enough apart that no
single read catches two of them:

```
hop 1   "the minister who restored the finances"  ->  Bhama Shah
hop 2   "the capital funded by that donation"     ->  Chavand
hop 3   "the year Pratap died there"              ->  1597
```

The point is that **hop 2's search term does not exist until hop 1 returns.**
You cannot embed a query for "the capital founded by the minister whose name you
don't know yet" — which is why this is recursion and not retrieval.

There's a planted trap in hop 1: the corpus spells it `Bhama Shah` (two words),
so grepping the modern `Bhamashah` returns zero hits. Zero matches means the
*pattern* is wrong, not that the fact is absent. Handling that is the difference
between a root that works and one that abstains.

## Typical result

```
answer        : 'Capital: Chavand; Pratap died there in 1597'
turns         : 6
sub-calls     : 3
chars read    : 119,791 of 3,000,265
coverage      : 3.99% of the document
root context  : 6,286 chars peak (0.210% of the corpus)
```

Three facts, 2.1 million characters apart, connected while the root held
**0.21%** of the corpus and read **4%** of it.

## The three ideas worth stealing

**1. The information bottleneck is `print()`.** `_exec` captures stdout rather
than streaming it. Whatever the root prints is its entire view of the document.
That's not a limitation working around — it's the mechanism.

**2. Hops connect through Python variables, not context.** The sub-model
answering hop 3 has never heard of Bhama Shah. Only short answer strings travel
back to the root, and those become the next search key.

**3. Endings must be earned.** Two guards, each protecting a different lie:

- `FINAL("1597")` — a **literal** answer is checked against everything the run
  actually observed *in prior turns*. `static_string` catches the three
  spellings of the same guess (`"1597"`, `f"1597"`, `"15" + "97"`). A value
  computed from variables is exempt: it came out of code that ran.
- `FINAL_NONE(...)` — rejected if `llm_query` was never called. Absence is the
  one claim REPL output can't ground, so it gets an *effort* test instead of a
  grounding test.

Grounding deliberately consults output from turns **before** the current one.
Appending first would let `print("1597"); FINAL("1597")` satisfy the guard with
its own guess — the model becoming its own evidence.

## What this leaves out

Deliberately, so the core stays readable: exec timeouts and per-example
deadlines, context eviction when the root's transcript outgrows its budget, a
`note()` scratchpad surviving eviction, retry on server overflow, repetition
breakers, and undersized-slice widening. Every one of those exists because a
real campaign hit it.
