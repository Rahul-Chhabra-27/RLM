"""Show the two endings being REJECTED, which is the interesting half.

An RLM that answers from parametric memory is not reading anything, and would
score well on famous facts while being useless on your own data. These two
guards are what force it to actually work.

    python3 demo_guards.py
"""

from __future__ import annotations

from . import corpus, llm
from .rlm import RLM


def canned(*replies: str):
    """A root that replays fixed turns. Not a model."""
    it = iter(replies)
    return lambda messages: next(it, "```python\nprint('no more script')\n```")


BANNER = "=" * 72


def main() -> None:
    document, offsets = corpus.build(300_000)
    sub = llm.extractive_sub()

    print(BANNER)
    print("GUARD 1 -- a literal answered from memory, having read nothing")
    print(BANNER)
    # A 4B model already knows Pratap died in 1597; it does not need the
    # document. That is exactly the failure a long-context benchmark must not
    # reward, because it scores well on famous facts and zero on your data.
    hallucinator = canned(
        'I already know this one.\n```python\nFINAL("1597")\n```',
        "Fine -- reading it properly.\n```python\n"
        "k = context.lower().rfind('chavand')\n"
        "ans = llm_query('In what year did the Rana die at Chavand?', context[k-6000:k+6000])\n"
        "print(ans)\n```",
        "Now it is grounded in something I read.\n```python\nFINAL(ans)\n```",
    )
    r = RLM(hallucinator, sub, chunk_chars=20_000, verbose=False).run(document, corpus.TASK)
    print(f"  turns       : {r.steps}   <- turn 1 rejected, so it had to go read")
    print(f"  sub-calls   : {r.sub_calls}")
    print(f"  answer      : {(r.answer or '')[:120]!r}")
    print("  The string '1597' was refused as a literal, then accepted once a")
    print("  sub-call actually produced it. Same characters, different provenance.\n")

    print(BANNER)
    print("GUARD 2 -- abstaining without having read anything")
    print(BANNER)
    quitter = canned(
        "The question text does not occur in the document.\n```python\n"
        "print(context.find('Name the capital Pratap founded'))\n"
        "FINAL_NONE('not present')\n```",
        "Rejected fairly. Searching for distinctive TERMS instead.\n```python\n"
        "m = re.search(r'bhama\\s*shah', context, re.I)\n"
        "print('found at', m.start())\n"
        "ans = llm_query('Which minister restored the finances, and what did it pay for?',\n"
        "                context[m.start()-6000:m.start()+6000])\n"
        "print(ans)\n```",
        "```python\nFINAL(ans)\n```",
    )
    r2 = RLM(quitter, sub, chunk_chars=20_000, verbose=False).run(document, corpus.TASK)
    print(f"  turns       : {r2.steps}   <- turn 1 abstention rejected")
    print(f"  sub-calls   : {r2.sub_calls}")
    print(f"  end         : {r2.end}")
    print(f"  answer      : {(r2.answer or '')[:120]!r}")
    print("  Searching for the QUESTION returned -1, as it always will: a natural-")
    print("  language question is not a string in any document. The guard refuses")
    print("  to let that count as evidence of absence.")


if __name__ == "__main__":
    main()
