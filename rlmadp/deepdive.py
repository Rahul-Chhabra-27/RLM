"""Trace the two mechanisms that ARE the RLM: creating the REPL, spawning a sub-call.

`tracing.py` answers "which function fired". This answers "what data crossed
which boundary, and what did each side actually see".

    python3 -m rlmadp.deepdive
"""

from __future__ import annotations

import sys

from . import corpus, llm
from .rlm import ROOT_SYSTEM_PROMPT, RLM

OUT = sys.__stdout__  # bypass _exec's stdout capture


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * 78, file=OUT)


class DeepRLM(RLM):
    def run(self, document, task):
        self._task = task           # _build_env needs it to render the prompt
        return super().run(document, task)

    def _build_env(self, document, state):
        rule("PHASE 1 - CREATING THE REPL  (rlm.py:202, once per run)")
        env = super()._build_env(document, state)

        print(f"the document is {len(document):,} chars", file=OUT)
        print("\nthe REPL namespace handed to every exec():", file=OUT)
        for k, v in env.items():
            if k == "__builtins__":
                kind = "<python builtins>"
            elif isinstance(v, str):
                kind = f"str, {len(v):,} chars"
            else:
                kind = type(v).__name__
            print(f"    {k:<12} {kind}", file=OUT)

        # The single most important fact about REPL creation: `context` is not a
        # copy. It is the SAME object, bound by reference. Creating the REPL
        # moves zero bytes -- which is why an RLM over a 10 MB document costs
        # the same to set up as one over 10 KB.
        same = env["context"] is document
        print(f"\n    env['context'] IS the document object: {same}"
              f"   (id {id(document)} == {id(env['context'])})", file=OUT)
        print("    -> building the REPL copied 0 bytes. Setup cost is independent", file=OUT)
        print("       of document size.\n", file=OUT)

        # run() builds the env FIRST, then renders the prompt (rlm.py:366, :356),
        # so the prompt analysis belongs here -- printing it from a run()
        # override would report phase 2 before phase 1.
        system = ROOT_SYSTEM_PROMPT.format(
            ctx_len=len(document), obs_limit=self.obs_limit,
            chunk_chars=self.chunk_chars, max_sub_calls=self.max_sub_calls,
            max_steps=self.max_steps, task=self._task)

        rule("PHASE 2 - WHAT THE ROOT IS TOLD  (rlm.py:356)")
        print(f"system prompt: {len(system):,} chars", file=OUT)
        print(f"document     : {len(document):,} chars", file=OUT)
        print(f"ratio        : the prompt is {len(system)/len(document):.4%} "
              "of the document\n", file=OUT)
        for probe in ["Haldighati", "Bhama Shah", "Chavand", "Sisodia", "1597"]:
            leak = probe in system and probe not in self._task
            print(f"    {probe:<12} in document: {str(probe in document):<5} "
                  f"in root prompt: {str(probe in system):<5}  "
                  f"[{'LEAK' if leak else 'ok'}]", file=OUT)
        print("\n    The ONLY thing the prompt says about the document is its LENGTH:", file=OUT)
        print(f"    \"It is {len(document):,} characters long. YOU CANNOT SEE IT.\"", file=OUT)
        print("    (Haldighati appears because it is in the TASK, not the corpus dump.)", file=OUT)
        return env


def spy_sub(real):
    """Wrap the sub-model and report both sides of the boundary."""
    n = [0]

    def sub(question, text):
        n[0] += 1
        rule(f"PHASE 3.{n[0]} - SPAWNING A SUB-CALL  (rlm.py:231 -> llm.py)")
        print("what the SUB-MODEL receives -- a fresh message list, nothing else:", file=OUT)
        print(f"    system   : 'Answer using ONLY the text provided...'", file=OUT)
        print(f"    question : {len(question):>7,} chars   {question[:62]!r}", file=OUT)
        print(f"    slice    : {len(text or ''):>7,} chars   {(text or '')[:62]!r}...", file=OUT)
        print("\n    NOT passed: the document, the root's transcript, any variable.", file=OUT)
        print("    The sub has no memory of earlier sub-calls. Fresh window every time.", file=OUT)
        answer = real(question, text)
        print(f"\nwhat comes BACK to the root:", file=OUT)
        print(f"    answer   : {len(answer):>7,} chars   {answer[:62]!r}...", file=OUT)
        shrink = len(text or "") / max(1, len(answer))
        print(f"\n    the root learns {len(answer)} chars about {len(text or ''):,} chars", file=OUT)
        print(f"    of document -- a {shrink:.0f}x reduction at the boundary.", file=OUT)
        return answer

    return sub


def main() -> None:
    document, offsets = corpus.build(3_000_000)
    print(f"corpus {len(document):,} chars   facts planted at "
          + ", ".join(f"{v:,}" for v in offsets.values()), file=OUT)

    agent = DeepRLM(llm.scripted_root(), spy_sub(llm.extractive_sub()), verbose=False)
    r = agent.run(document, corpus.TASK)

    rule("RESULT")
    print(f"answer        : {r.answer!r}", file=OUT)
    print(f"sub-calls     : {r.sub_calls}", file=OUT)
    print(f"chars read    : {r.chars_read:,} of {len(document):,} ({r.coverage:.2%})", file=OUT)
    print(f"root context  : {r.root_peak_chars:,} chars peak "
          f"({r.root_peak_chars/len(document):.3%} of the corpus)", file=OUT)


if __name__ == "__main__":
    main()
