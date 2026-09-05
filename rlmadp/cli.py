"""Run the RLM on the synthetic Rajputana corpus.

    python3 run.py                      # offline trace, scripted root
    python3 run.py --vllm               # real model on localhost:8000
    python3 run.py --vllm --chars 8000  # starve the slice and watch it fail
"""

from __future__ import annotations

import argparse

from . import corpus, llm
from .rlm import RLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", action="store_true", help="use a real OpenAI-compatible server")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="root model: writes the REPL code, needs instruction-following")
    ap.add_argument("--sub-model", default=None,
                    help="sub model (default: same as --model). Only reads one slice "
                         "and answers one question, so a smaller one is usually fine")
    ap.add_argument("--size", type=int, default=3_000_000, help="corpus size in characters")
    ap.add_argument("--chars", type=int, default=40_000, help="max chars per llm_query slice")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="root completion budget; too low truncates its code block")
    ap.add_argument("--sub-calls", type=int, default=8)
    args = ap.parse_args()

    document, offsets = corpus.build(args.size)
    print(f"corpus: {len(document):,} chars")
    for name, off in offsets.items():
        print(f"  planted {name:20s} at {off:>10,}  ({off / len(document):5.1%})")
    print(f"\ntask: {corpus.TASK}\n" + "=" * 72)

    if args.vllm:
        root, sub = llm.openai_compatible(args.base_url, args.model,
                                          max_tokens=args.max_tokens,
                                          sub_model=args.sub_model)
    else:
        print("[offline mode: scripted root + extractive sub -- machinery only]")
        print("[the root REPLAYS a fixed transcript, so it cannot react to a failed]")
        print("[sub-call. Watch 'NOT FOUND' below: use --vllm to see real recovery.]\n")
        root, sub = llm.scripted_root(), llm.extractive_sub()

    agent = RLM(root, sub, max_steps=args.steps, max_sub_calls=args.sub_calls,
                chunk_chars=args.chars)
    result = agent.run(document, corpus.TASK)

    print("\n" + "=" * 72)
    print(f"answer        : {result.answer!r}")
    if result.abstained:
        print(f"abstained     : {result.reason}")
    print(f"end           : {result.end}")
    print(f"turns         : {result.steps}")
    print(f"sub-calls     : {result.sub_calls} "
          f"({result.sub_not_found} came back NOT FOUND)")
    print(f"chars read    : {result.chars_read:,} of {len(document):,}")
    print(f"coverage      : {result.coverage:.2%} of the document")
    # THE number. The root's own context never grew with the document -- this is
    # the entire claim of the method, stated as a measurement.
    print(f"root context  : {result.root_peak_chars:,} chars peak "
          f"({result.root_peak_chars / len(document):.3%} of the corpus)")


if __name__ == "__main__":
    main()
