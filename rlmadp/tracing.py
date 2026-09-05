"""Log every trigger point in rlm.py, in the order it actually fires.

Wraps the real functions -- nothing here reimplements anything. Run it and the
output IS the call order:

    python3 trace.py
"""

from __future__ import annotations

import sys

from . import corpus, llm
from . import rlm as rlm_mod
from .rlm import RLM

STEP = [0]


def log(where: str, line: int, what: str) -> None:
    STEP[0] += 1
    # Write to the REAL stdout: _exec redirects stdout to capture the model's
    # prints, and a log line written there would be swallowed into the
    # observation instead of shown -- which is exactly what happened first try.
    print(f"{STEP[0]:>3}. rlm.py:{line:<3}  {where:<22} {what}", file=sys.__stdout__)


class TracedRLM(RLM):
    """Same behaviour, every entry point announced."""

    def _build_env(self, document, state):
        log("_build_env", 202, "build the REPL namespace (once per run)")
        env = super()._build_env(document, state)

        raw_q, raw_f, raw_n = env["llm_query"], env["FINAL"], env["FINAL_NONE"]

        def llm_query(question, text=None):
            size = 0 if text is None else len(text)
            log("llm_query", 231, f"sub-call requested, slice={size:,} chars")
            before = state["sub_calls"], state["cache_hits"]
            out = raw_q(question, text)
            if state["cache_hits"] > before[1]:
                log("llm_query", 246, "CACHE HIT -- no model called")
            elif state["sub_calls"] > before[0]:
                log("  _record_span", 211, f"coverage now {state['coverage']:.2%}")
                log("llm_query", 286, f"sub returned {len(out)} chars")
            else:
                log("llm_query", 250, "REFUSED (budget/limit) -- returned a notice")
            return out

        def FINAL(v):
            log("FINAL", 301, f"root commits: {str(v)[:46]!r}")
            return raw_f(v)

        def FINAL_NONE(reason=""):
            log("FINAL_NONE", 305, f"root abstains: {reason[:40]!r}")
            return raw_n(reason)

        env.update({"llm_query": llm_query, "FINAL": FINAL, "FINAL_NONE": FINAL_NONE})
        return env

    def _exec(self, code, env):
        log("_exec", 325, "run the code block, capture stdout")
        return super()._exec(code, env)

    @staticmethod
    def _grounded(value, seen, task):
        ok = RLM._grounded(value, seen, task)
        log("_grounded", 446, f"is {value[:28]!r} in prior output? -> {ok}")
        return ok


_real_literals = rlm_mod.final_literals


def traced_literals(code):
    out = _real_literals(code)
    if "FINAL" in code:
        log("final_literals", 138, f"AST scan -> literals={out or '{} (computed)'}")
    return out


rlm_mod.final_literals = traced_literals


def main() -> None:
    document, offsets = corpus.build(3_000_000)
    print(f"document: {len(document):,} chars   task: {corpus.TASK[:58]}...")
    for k, v in offsets.items():
        print(f"  planted {k:20s} at {v:>10,}")
    print("\n" + "-" * 78)
    log("run", 343, "loop starts; system prompt built from the template")
    r = TracedRLM(llm.scripted_root(), llm.extractive_sub(), verbose=False).run(
        document, corpus.TASK
    )
    print("-" * 78)
    print(f"end={r.end}  answer={r.answer!r}")
    print(f"turns={r.steps}  sub_calls={r.sub_calls}  coverage={r.coverage:.2%}  "
          f"root_peak={r.root_peak_chars:,} chars")


if __name__ == "__main__":
    main()
