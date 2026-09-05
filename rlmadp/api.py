"""The paper's framing: an RLM is a drop-in replacement for a chat completion.

    Zhang, Kraska & Khattab, "Recursive Language Models", arXiv:2512.24601:
    `rlm.completion(messages)` substitutes for `gpt5.completion(messages)`.

That interface is the point. The caller passes an ordinary message list whose
content happens to be enormous; the RLM decides how to read it. Nothing about
the call site says "long context".

    from rlmadp import api
    text = api.completion(messages, base_url=..., model=...)
"""

from __future__ import annotations

from typing import Callable, Optional

from . import llm as _llm
from .rlm import RLM, RLMResult

# Below this, an RLM is pure overhead: the model can simply read the prompt.
# The paper's whole argument is about prompts that do not fit; a scaffold that
# fires on a 2 KB message would add turns and cost for nothing.
DIRECT_THRESHOLD_CHARS = 20_000


def split_messages(messages: list[dict]) -> tuple[str, str]:
    """Separate the bulk (the 'context') from the instruction (the 'query').

    Heuristic, and deliberately simple: the single largest message is the
    context, everything else concatenated is the task. Real callers put a 3 MB
    document in one message and a question in another, which is exactly this
    shape. Callers who know better should use RLM directly.
    """
    if not messages:
        return "", ""
    biggest = max(range(len(messages)), key=lambda i: len(messages[i].get("content", "")))
    context = messages[biggest].get("content", "")
    task = "\n\n".join(
        m.get("content", "") for i, m in enumerate(messages) if i != biggest
    ).strip()
    return context, task or "Answer the question in the provided text."


def completion(
    messages: list[dict],
    *,
    base_url: str = "http://localhost:8000/v1",
    model: str = "Qwen/Qwen3-4B-Instruct-2507",
    sub_model: Optional[str] = None,
    root: Optional[Callable] = None,
    sub: Optional[Callable] = None,
    threshold: int = DIRECT_THRESHOLD_CHARS,
    return_result: bool = False,
    **rlm_kwargs,
) -> str | RLMResult:
    """Answer `messages`, recursing only if the prompt warrants it."""
    if root is None or sub is None:
        root, sub = _llm.openai_compatible(base_url, model, sub_model=sub_model)

    context, task = split_messages(messages)
    if len(context) <= threshold:
        # Short enough to read directly -- no REPL, no recursion, one call.
        answer = sub(task, context)
        if return_result:
            return RLMResult(answer=answer, end="direct", steps=0)
        return answer

    result = RLM(root, sub, **rlm_kwargs).run(context, task)
    if return_result:
        return result
    return result.answer if result.answer is not None else f"NOT FOUND: {result.reason}"
