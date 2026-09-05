# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "rlmadp @ git+https://github.com/Rahul-Chhabra-27/RLM.git",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Recursive Language Models, in one notebook

    Long-context models don't fail because the prompt exceeds the window. They
    fail *inside* it — accuracy decays well before the hard limit as attention
    spreads thin. A bigger window moves the cliff; it doesn't remove it.

    An RLM stops feeding the model the document at all:

    > Bind the document to a Python variable. Give the model a REPL. Let it write
    > code to find what it needs, and spawn **fresh** sub-models on the slices it
    > selects.

    The root model holds only its own short transcript, so document length never
    enters its context. Below, a root answers a 3-hop question over a 3 MB
    Rajputana chronicle while holding **~0.2%** of it.

    Everything here is CPU-only and dependency-free — the RLM core uses nothing
    but `ast`, `io`, `re`, `contextlib` and `dataclasses`.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    from rlmadp import RLM
    from rlmadp import corpus as corpus_mod
    from rlmadp import llm
    return RLM, corpus_mod, llm, mo


@app.cell(hide_code=True)
def _(mo):
    size = mo.ui.slider(
        start=200_000, stop=5_000_000, step=200_000, value=3_000_000,
        label="corpus size (characters)", show_value=True,
    )
    chars = mo.ui.slider(
        start=2_000, stop=80_000, step=2_000, value=40_000,
        label="max chars per llm_query slice", show_value=True,
    )
    mo.vstack([size, chars])
    return chars, size


@app.cell(hide_code=True)
def _(corpus_mod, mo, size):
    document, offsets = corpus_mod.build(size.value)
    _rows = "\n".join(
        f"| {k} | {v:,} | {v / len(document):.1%} |" for k, v in offsets.items()
    )
    mo.md(
        f"""
    ## The haystack

    **{len(document):,} characters** of generated chronicle. Three facts are
    planted far enough apart that no single read catches two of them:

    | fact | offset | position |
    |---|---:|---:|
    {_rows}

    **Task —** {corpus_mod.TASK}

    The chain: *the minister* → **Bhama Shah** → *the capital he funded* →
    **Chavand** → *the year Pratap died there* → **1597**.

    Hop 2's search term does not exist until hop 1 returns. You cannot embed a
    query for "the capital founded by the minister whose name you don't know
    yet" — which is why this is recursion and not retrieval.

    There's a trap in hop 1: the chronicle spells it `Bhama Shah` (two words), so
    grepping the modern `Bhamashah` returns zero hits.
    """
    )
    return (document,)


@app.cell(hide_code=True)
def _(RLM, chars, corpus_mod, document, llm):
    # The scripted root replays a fixed transcript and the extractive sub is a
    # keyword sentence-picker: neither is a language model. They make the
    # MACHINERY runnable with no GPU and no API key. For real behaviour, point
    # llm.openai_compatible() at a served model (see the last cell).
    _agent = RLM(
        llm.scripted_root(),
        llm.extractive_sub(),
        chunk_chars=chars.value,
        verbose=False,
    )
    result = _agent.run(document, corpus_mod.TASK)
    return (result,)


@app.cell(hide_code=True)
def _(document, mo, result):
    _verdict = (
        f"### ✅ `{result.answer}`"
        if result.answer
        else f"### ⚠️ abstained — *{result.reason}*"
    )
    mo.md(
        f"""
    ## Result

    {_verdict}

    | | |
    |---|---:|
    | turns | {result.steps} |
    | sub-calls | {result.sub_calls} ({result.sub_not_found} returned NOT FOUND) |
    | characters read | {result.chars_read:,} of {len(document):,} |
    | document coverage | {result.coverage:.2%} |
    | **root context peak** | **{result.root_peak_chars:,} chars
      ({result.root_peak_chars / len(document):.3%} of the corpus)** |

    Drag the slice slider down. Below ~24,000 characters the window stops
    reaching the planted passage, hop 1 fails, the chain breaks — and the run
    **abstains** instead of inventing an answer.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, result):
    _turns = "\n\n---\n\n".join(
        f"**Turn {t.n}**\n\n```python\n{t.code}\n```\n\n"
        f"```\nREPL: {t.observation[:600]}\n```"
        for t in result.turns
    )
    mo.md(f"## The transcript\n\n{_turns}")
    return


@app.cell(hide_code=True)
def _(RLM, corpus_mod, document, llm, mo):
    # Sweep the one knob that matters. Each row is a full independent run.
    _rows = []
    for _c in (3_000, 6_000, 12_000, 24_000, 40_000, 80_000):
        _r = RLM(
            llm.scripted_root(), llm.extractive_sub(),
            chunk_chars=_c, verbose=False,
        ).run(document, corpus_mod.TASK)
        _rows.append(
            f"| {_c:,} | {'found' if _r.answer else '**abstained**'} "
            f"| {_r.coverage:.2%} | {_r.sub_calls} | {_r.end} |"
        )
    mo.md(
        "## Slice-size sweep\n\n"
        "| max chars | outcome | coverage | sub-calls | end |\n"
        "|---:|---|---:|---:|---|\n" + "\n".join(_rows) + "\n\n"
        "Above 40,000 nothing improves: the root's windows *are* 40,000 wide. "
        "A cap is a ceiling the root may use, not a floor that makes it read."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Using a real model

    `llm.openai_compatible()` speaks to any OpenAI-compatible endpoint over
    `urllib` — vLLM, Ollama, anything. molab has no GPU by default, so point it
    at a server you already run:

    ```python
    root, sub = llm.openai_compatible(
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen3-4B-Instruct-2507",
    )
    result = RLM(root, sub, chunk_chars=40_000).run(document, corpus_mod.TASK)
    ```

    Source and the cluster runbook: <https://github.com/Rahul-Chhabra-27/RLM>
    """
    )
    return


if __name__ == "__main__":
    app.run()
