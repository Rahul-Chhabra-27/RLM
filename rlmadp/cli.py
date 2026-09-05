"""Run the RLM, optionally against the vanilla long-context baseline.

    python3 -m rlmadp.cli                      # offline trace, hop task
    python3 -m rlmadp.cli --task count         # the sweep task
    python3 -m rlmadp.cli --vllm --compare     # RLM vs vanilla, real model
    python3 -m rlmadp.cli --vllm --max-depth 2 # allow nested RLMs
"""

from __future__ import annotations

import argparse

from . import baseline, corpus, llm
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
    ap.add_argument("--task", choices=["hop", "count"], default="hop",
                    help="hop: 3-hop chain, findable by search. "
                         "count: whole-document aggregation, must be swept")
    ap.add_argument("--size", type=int, default=3_000_000, help="corpus size in characters")
    ap.add_argument("--chars", type=int, default=40_000, help="max chars per llm_query slice")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--sub-calls", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3072,
                    help="root completion budget; too low truncates its code block")
    ap.add_argument("--max-depth", type=int, default=1,
                    help="how deep llm_query(recurse=True) may nest. The paper "
                         "runs depth 1 in all experiments")
    ap.add_argument("--compare", action="store_true",
                    help="also run the vanilla long-context baseline on the same task")
    ap.add_argument("--baseline-chars", type=int, default=60_000,
                    help="how much of the document the baseline may read")
    args = ap.parse_args()

    if args.task == "count":
        document, truth = corpus.build_aggregation(args.size)
        task = corpus.TASK_COUNT
        print(f"corpus: {len(document):,} chars   ground truth = {truth}")
    else:
        document, offsets = corpus.build(args.size)
        task, truth = corpus.TASK, None
        print(f"corpus: {len(document):,} chars")
        for name, off in offsets.items():
            print(f"  planted {name:20s} at {off:>10,}  ({off / len(document):5.1%})")
    print(f"\ntask: {task}\n" + "=" * 72)

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
                chunk_chars=args.chars, max_depth=args.max_depth)
    result = agent.run(document, task)

    print("\n" + "=" * 72)
    print(f"answer        : {result.answer!r}")
    if result.abstained:
        print(f"abstained     : {result.reason}")
    if truth is not None:
        print(f"ground truth  : {truth}")
    print(f"end           : {result.end}")
    print(f"turns         : {result.steps}")
    print(f"sub-calls     : {result.sub_calls} "
          f"({result.sub_not_found} came back NOT FOUND"
          + (f", {result.recursive_calls} recursive" if result.recursive_calls else "")
          + ")")
    print(f"chars read    : {result.chars_read:,} of {len(document):,}")
    print(f"coverage      : {result.coverage:.2%} of the document")
    # THE number. The root's own context never grew with the document -- this is
    # the entire claim of the method, stated as a measurement.
    print(f"root context  : {result.root_peak_chars:,} chars peak "
          f"({result.root_peak_chars / len(document):.3%} of the corpus)")

    if args.compare:
        print("\n" + "=" * 72)
        print("VANILLA BASELINE -- same task, document read natively")
        print("=" * 72)
        for keep in ("head", "middle", "tail"):
            b = baseline.vanilla_answer(sub, document, task,
                                        max_chars=args.baseline_chars, keep=keep)
            print(f"  keep={keep:<7} retained {b.context_retained:6.2%}  "
                  f"answer: {(b.answer or b.end)[:60]!r}")
        print("\n  Truncation is the baseline's whole problem and the RLM's whole point:")
        print("  which fragment survives decides the answer. Report `retained` beside")
        print("  any baseline score -- a low one may be a property of the window.")


if __name__ == "__main__":
    main()
