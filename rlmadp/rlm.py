"""A minimal Recursive Language Model (RLM).

The whole idea in one sentence:

    Do not put the document in the model's context window.
    Put it in a Python variable, and let the model write code to read it.

A *root* model is given a REPL in which the document is bound to `context`.
It never sees the text. It can only:

    - run Python against it   (find, regex, slicing, counting)  -- free
    - call `llm_query(...)`   (a FRESH sub-model on a substring) -- costs
    - call `FINAL(...)`       (commit to an answer)

Because the root only ever holds its own short transcript, its context stays
~2-6k tokens whether the document is 100KB or 100MB. That is the point: the
model's accuracy stops degrading with document length, because document length
stops entering its context at all.

Read the file in this order:
    1. ROOT_SYSTEM_PROMPT  -- everything the root knows about its world
    2. RLM._build_env      -- the tools it gets, especially llm_query
    3. RLM.run             -- the loop that ties them together
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# The root replies in prose + fenced code. We execute only the code.
CODE_BLOCK = re.compile(r"```(?:python)?[ \t]*\n(.*?)```", re.DOTALL)


ROOT_SYSTEM_PROMPT = """\
You are a reasoning agent driving a persistent Python REPL.

A document has been loaded into the REPL as a variable called `context`.
It is {ctx_len:,} characters long. YOU CANNOT SEE IT. The only way to learn
anything about it is to run code.

Reply with ONE fenced Python code block per turn. It is executed, and whatever
you print comes back to you in the next message (truncated at {obs_limit:,}
characters). Variables persist across turns.

TOOLS
    context : str
        The full document. Slice it, search it, count it -- all free.

    llm_query(question: str, text: str) -> str
        Ask a FRESH sub-model a question about a piece of text. The sub-model
        CANNOT see `context` or any of your variables -- it sees ONLY the two
        strings you pass. Pass the question first and the document slice
        second:  ans = llm_query("Who founded X?", context[i:j])
        Each slice may be up to {chunk_chars:,} characters. Prefer FEW, LARGE
        slices over many small ones: a slice of {chunk_chars:,} chars costs one
        call, and so does a slice of 500. You have {max_sub_calls} calls total.

    FINAL(value)
        Commit to the answer and stop. The value must be something your code
        actually COMPUTED or something you actually SAW in REPL output.
        FINAL with a guessed literal will be REJECTED.

    FINAL_NONE(reason)
        The document genuinely does not contain the answer. Legitimate, and
        always better than guessing -- but rejected if you have not called
        llm_query at least once.

HOW TO SEARCH
    Never search for the question itself. A natural-language question appears
    nowhere in any document as a literal string. Pull out the DISTINCTIVE
    terms -- proper nouns, names, titles, years, rare words -- and search for
    each one SEPARATELY and case-insensitively:

        idx = context.lower().find(term.lower())

    Zero matches usually means your PATTERN is wrong, not that the fact is
    absent. Historical texts use spelling variants. Try `re.finditer` with a
    loose pattern, print the text around a near-miss to see the real format,
    then retry.

MULTI-HOP
    If the answer requires chaining facts, do them one at a time. The answer
    to hop 1 becomes the SEARCH TERM for hop 2. You cannot search for hop 2
    until hop 1 has come back.

You have at most {max_steps} turns. Be efficient.

THE TASK:
{task}
"""

REJECT_UNGROUNDED = (
    "REJECTED: FINAL({value!r}) -- that value never appeared in any REPL output, "
    "so it looks like a guess from memory rather than something you read in the "
    "document. Go find it in `context` and print it first."
)

REJECT_LAZY_ABSTAIN = (
    "REJECTED: you called FINAL_NONE without ever calling llm_query, so you have "
    "not actually read any of the document. A failed search for the whole question "
    "proves nothing. Extract distinctive terms and search for each separately."
)

NO_CODE_NUDGE = (
    "No code block found. Reply with exactly one fenced Python block, e.g.\n"
    "```python\nprint(len(context))\n```"
)


def static_string(node: ast.AST) -> Optional[str]:
    """The value of `node` if it is knowable WITHOUT running any code, else None.

    This is what separates a guessed answer from a computed one. FINAL(answer)
    where `answer` is a variable is grounded by construction -- the value came
    out of code that actually ran. FINAL("1597") is a claim from the model's
    memory, and must be checked.

    Handles the three spellings of the same guess: "1597", f"1597", "15" + "97".
    Matching only ast.Constant would reject the honest spelling and wave the
    evasive ones straight through.
    """
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):  # f-string, static only if nothing interpolated
        parts = [static_string(v) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        lhs, rhs = static_string(node.left), static_string(node.right)
        return lhs + rhs if lhs is not None and rhs is not None else None
    return None


def final_literals(code: str) -> set[str]:
    """Every FINAL(...) argument in `code` that is a hard-coded literal."""
    out: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "FINAL" and node.args):
            lit = static_string(node.args[0])
            if lit is not None:
                out.add(lit)
    return out


@dataclass
class Turn:
    """One round trip: what the root wrote, and what the REPL said back."""

    n: int
    code: str
    observation: str


@dataclass
class RLMResult:
    answer: Optional[str] = None
    abstained: bool = False
    reason: str = ""
    end: str = "ok"  # ok | abstained | out_of_steps | error
    steps: int = 0
    sub_calls: int = 0
    sub_not_found: int = 0
    chars_read: int = 0
    coverage: float = 0.0
    root_peak_chars: int = 0
    turns: list[Turn] = field(default_factory=list)


class RLM:
    def __init__(
        self,
        root: Callable[[list[dict]], str],
        sub: Callable[[str, Optional[str]], str],
        *,
        max_steps: int = 12,
        max_sub_calls: int = 8,
        chunk_chars: int = 40_000,
        obs_limit: int = 2_000,
        verbose: bool = True,
    ):
        # `root` and `sub` are just functions. Anything that maps messages ->
        # string works: an OpenAI client, a local vLLM server, a fake. Keeping
        # them as plain callables is what makes this file testable offline.
        self.root = root
        self.sub = sub
        self.max_steps = max_steps
        self.max_sub_calls = max_sub_calls
        self.chunk_chars = chunk_chars
        self.obs_limit = obs_limit
        self.verbose = verbose

    # ------------------------------------------------------------------ env
    def _build_env(self, document: str, state: dict) -> dict:
        """Construct the REPL namespace handed to the root's code.

        Everything the root can do lives in this dict. If it isn't here, the
        root cannot reach it -- which is the sandbox, such as it is.
        """
        cache: dict[tuple[str, Optional[str]], str] = {}
        spans: list[tuple[int, int]] = []

        def _record_span(start: int, end: int) -> None:
            """Track which parts of the document were actually READ.

            Union of spans, not sum: two overlapping reads of the same passage
            do not count twice. `coverage` is then an honest 'how much of the
            haystack did this run actually look at'.
            """
            spans.append((start, end))
            merged, cur_s, cur_e = 0, -1, -1
            for s, e in sorted(spans):
                if s > cur_e:
                    if cur_e >= 0:
                        merged += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            if cur_e >= 0:
                merged += cur_e - cur_s
            state["coverage"] = merged / len(document) if document else 0.0

        def llm_query(question: str, text: Optional[str] = None) -> str:
            """Spawn a fresh sub-model on a slice of the document.

            THIS is the recursion. Note what it does NOT do: it does not pass
            `document`, the root's transcript, or any variable. The sub-model
            gets two strings and nothing else, so its context window is clean
            no matter how long the run has been going.
            """
            question = str(question)
            text = None if text is None else str(text)

            # Cache on the FULL pair. Keying on the truncated slice would make
            # context[0:100000] and context[0:200000] collide -- silently
            # returning one region's answer for a different region.
            key = (question, text)
            if key in cache:
                state["cache_hits"] += 1
                return cache[key]

            if state["sub_calls"] >= self.max_sub_calls:
                # Degrade to a message rather than raising: the root can still
                # finish from what it has already seen.
                return (
                    "[SUB-CALL LIMIT REACHED] No further llm_query calls are "
                    "available. Answer from what you have already seen, or use "
                    "plain string operations on `context`."
                )

            # Fit question + slice into ONE budget. The question is an
            # instruction, not the payload, so it gets a quarter; the slice
            # takes the rest. Capping each side separately would let a call
            # carry ~2x what the prompt advertised.
            truncated = False
            if text is None:
                if len(question) > self.chunk_chars:
                    question, truncated = question[: self.chunk_chars], True
            else:
                qcap = max(1, self.chunk_chars // 4)
                if len(question) > qcap:
                    question, truncated = question[:qcap], True
                room = max(0, self.chunk_chars - len(question))
                if len(text) > room:
                    text, truncated = text[:room], True
            if truncated:
                question += f"\n[NOTE: truncated at {self.chunk_chars} chars total]"

            # Where in the document did this slice come from? Only findable if
            # the root SLICED it. A slice the root assembled by hand (an
            # f-string, concatenated fragments) has no position, so it is not
            # counted toward coverage rather than guessed at.
            if text:
                start = document.find(text)
                if start >= 0:
                    _record_span(start, start + len(text))
                state["chars_read"] += len(text)

            answer = self.sub(question, text)
            state["sub_calls"] += 1
            # A sub-call that read the WRONG region is not an error -- it is the
            # normal cost of searching. But a run whose answer is right while
            # most sub-calls came back empty did not earn that answer, so count
            # them rather than letting the final score hide them.
            if answer.strip().upper().startswith("NOT FOUND"):
                state["sub_not_found"] += 1
            cache[key] = answer
            if self.verbose:
                size = len(text) if text else 0
                print(f"      [sub-call {state['sub_calls']}: {size:,} chars -> {answer[:70]!r}]")
            return answer

        def FINAL(value: Any) -> None:
            state["final"] = str(value)
            state["done"] = True

        def FINAL_NONE(reason: str = "") -> None:
            # Deliberately separate from FINAL so the grounding check cannot
            # reject it. "The answer is not here" is by construction absent
            # from the REPL output, so the honest ending would otherwise look
            # exactly like a hallucinated guess.
            state["final"] = None
            state["reason"] = str(reason).strip()
            state["abstained"] = True
            state["done"] = True

        return {
            "context": document,
            "llm_query": llm_query,
            "FINAL": FINAL,
            "FINAL_NONE": FINAL_NONE,
            "re": re,
            "__builtins__": __builtins__,
        }

    # ------------------------------------------------------------------ exec
    def _exec(self, code: str, env: dict) -> str:
        """Run one code cell and return what the root gets to see.

        stdout is captured, not streamed: the root's whole view of the document
        is whatever it chose to print, so this buffer IS the information
        bottleneck the design depends on.
        """
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, env)  # noqa: S102 -- executing model code is the point
        except Exception as exc:
            out = buf.getvalue()
            return (out + f"\n{type(exc).__name__}: {exc}").strip()
        out = buf.getvalue().strip()
        return out or "[no output]"

    # ------------------------------------------------------------------- run
    def run(self, document: str, task: str) -> RLMResult:
        state = {
            "final": None,
            "reason": "",
            "done": False,
            "abstained": False,
            "sub_calls": 0,
            "sub_not_found": 0,
            "cache_hits": 0,
            "chars_read": 0,
            "coverage": 0.0,
        }
        env = self._build_env(document, state)

        system = ROOT_SYSTEM_PROMPT.format(
            ctx_len=len(document),
            obs_limit=self.obs_limit,
            chunk_chars=self.chunk_chars,
            max_sub_calls=self.max_sub_calls,
            max_steps=self.max_steps,
            task=task,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Begin. Write your first code block."},
        ]

        result = RLMResult()
        # Everything the run has ever OBSERVED. Never truncated, never
        # compacted -- it is the evidence base the grounding check consults.
        seen = ""

        for step in range(1, self.max_steps + 1):
            result.steps = step
            reply = self.root(messages)
            messages.append({"role": "assistant", "content": reply})
            result.root_peak_chars = max(
                result.root_peak_chars, sum(len(m["content"]) for m in messages)
            )

            match = CODE_BLOCK.search(reply)
            if not match:
                messages.append({"role": "user", "content": NO_CODE_NUDGE})
                continue

            code = match.group(1)
            if self.verbose:
                print(f"\n--- turn {step} ---\n{code.strip()}")

            literals = final_literals(code)
            observation = self._exec(code, env)
            # Grounding consults output from turns BEFORE this one. Appending
            # first would let `print("1597"); FINAL("1597")` satisfy the guard
            # with its own guess -- the model would be its own evidence.
            prior_seen = seen
            seen += "\n" + observation

            # --- the two endings, each with its own guard --------------------
            if state["done"]:
                if state["abstained"]:
                    if state["sub_calls"] == 0:
                        state["done"] = state["abstained"] = False
                        messages.append({"role": "user", "content": REJECT_LAZY_ABSTAIN})
                        continue
                    result.abstained = True
                    result.reason = state["reason"]
                    result.end = "abstained"
                    break

                value = state["final"] or ""
                # Only LITERAL answers are checked. FINAL(answer) with a
                # variable is grounded by construction; FINAL("1597") is a
                # claim from parametric memory and must be found in something
                # the run actually read. Word boundaries, not containment --
                # containment whitelists every short answer, since a document
                # mentioning "12" anywhere would license the answer 12.
                if value in literals and not self._grounded(value, prior_seen, task):
                    state["done"], state["final"] = False, None
                    messages.append(
                        {"role": "user", "content": REJECT_UNGROUNDED.format(value=value)}
                    )
                    continue
                result.answer = value
                result.end = "ok"
                break

            shown = observation[: self.obs_limit]
            if len(observation) > self.obs_limit:
                shown += f"\n[...truncated, {len(observation):,} chars total]"
            if self.verbose:
                print(f"REPL: {shown}")
            messages.append({"role": "user", "content": f"REPL output:\n{shown}"})
            result.turns.append(Turn(step, code.strip(), shown))
        else:
            result.end = "out_of_steps"

        result.sub_calls = state["sub_calls"]
        result.sub_not_found = state["sub_not_found"]
        result.chars_read = state["chars_read"]
        result.coverage = state["coverage"]
        return result

    @staticmethod
    def _grounded(value: str, seen: str, task: str) -> bool:
        """Did every substantive token of the answer appear in something read?"""
        tokens = [t for t in re.findall(r"[A-Za-z0-9']+", value) if len(t) > 2]
        if not tokens:
            return bool(value.strip())
        hay = (seen + " " + task).lower()
        return all(re.search(rf"\b{re.escape(t.lower())}\b", hay) for t in tokens)
