"""Show the actual slice geometry: where each sub-call cut, and why.

`deepdive.py` shows what crosses the boundary. This shows WHERE the boundary
was drawn -- the exact character ranges, what sat at their edges, where the
planted fact fell inside them, and which earlier answer supplied the index.

    python3 -m rlmadp.slices
"""

from __future__ import annotations

import sys

from . import corpus, llm
from .rlm import RLM

OUT = sys.__stdout__


def bar(doc_len: int, marks: dict[str, int], width: int = 66) -> str:
    """One-line map of the document with the planted facts marked."""
    row = ["-"] * width
    labels = [" "] * width
    for name, off in marks.items():
        col = min(width - 1, int(off / doc_len * width))
        row[col] = "|"
        tag = name.split()[0]
        for k, ch in enumerate(tag):
            if col + k < width:
                labels[col + k] = ch
    return "".join(row) + "\n" + "".join(labels)


def show(doc: str, n: int, question: str, text: str, answer: str,
         trigger: str, prev_term: str | None) -> None:
    start = doc.find(text)
    end = start + len(text)
    print(f"\n\033[1mSUB-CALL {n}\033[0m", file=OUT)
    print(f"  index came from : {trigger}", file=OUT)
    if prev_term:
        print(f"  search term     : {prev_term!r}  <- produced by sub-call {n - 1}", file=OUT)
    print(f"  slice           : context[{start:,} : {end:,}]   ({len(text):,} chars)", file=OUT)
    print(f"  = {start / len(doc):.1%} to {end / len(doc):.1%} of the document", file=OUT)

    # Where does the planted fact actually sit inside this window?
    for label, needle in (("hop1", "Bhama Shah came to him"),
                          ("hop2", "His seat during this period"),
                          ("hop3", "nineteenth of January 1597")):
        at = doc.find(needle)
        if start <= at < end:
            off = at - start
            print(f"  fact ({label})     : at doc {at:,} = {off:,} chars into the "
                  f"slice ({off / len(text):.0%} through it)", file=OUT)

    print(f"  slice STARTS    : ...{text[:66].strip()!r}", file=OUT)
    print(f"  slice ENDS      : {text[-66:].strip()!r}...", file=OUT)
    print(f"  question        : {question[:70]!r}", file=OUT)
    print(f"  answer          : {answer[:70]!r}  ({len(answer)} chars)", file=OUT)


def main() -> None:
    doc, offsets = corpus.build(1_000_000)
    print(f"\033[1mDOCUMENT\033[0m  {len(doc):,} chars", file=OUT)
    print(bar(len(doc), offsets), file=OUT)
    for k, v in offsets.items():
        print(f"  {k:22s} {v:>9,}   ({v / len(doc):5.1%})", file=OUT)

    calls: list[tuple] = []
    real_sub = llm.extractive_sub()

    def spy(question, text):
        ans = real_sub(question, text)
        calls.append((question, text, ans))
        return ans

    RLM(llm.scripted_root(), spy, chunk_chars=40_000, verbose=False).run(doc, corpus.TASK)

    # What the root learned from each call, and used to place the next slice.
    triggers = [
        're.search(r"bhama\\s*shah") -> first hit',
        're.finditer(r"bhama\\s*shah") -> hits[1], the SECOND occurrence',
        're.finditer("chavand") -> hits[-1], the LAST occurrence',
    ]
    terms = [None, "Bhama Shah", "Chavand"]
    for i, (q, t, a) in enumerate(calls):
        show(doc, i + 1, q, t, a, triggers[i], terms[i])

    print("\n\033[1mTHE CHAIN\033[0m", file=OUT)
    print("  task names nobody: 'a minister' / 'the capital' / 'the year'", file=OUT)
    prev = "(from the task wording)"
    for i, (q, t, a) in enumerate(calls):
        s = doc.find(t)
        gained = ["Bhama Shah", "Chavand", "1597"][i]
        print(f"    search {prev:<28} -> slice @ {s:>7,} -> learned {gained!r}", file=OUT)
        prev = f"{gained!r}"
    print("\n  Each slice is placed by a term the PREVIOUS answer produced.", file=OUT)
    print("  None of those terms is in the task, so the order cannot be changed.", file=OUT)


if __name__ == "__main__":
    main()
