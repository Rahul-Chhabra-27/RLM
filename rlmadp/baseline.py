"""Vanilla long-context baseline -- the thing an RLM has to beat.

The paper's claim is comparative: an RLM outperforms a frontier model reading
the prompt natively. Without this baseline you can measure that an RLM works,
but not that it is worth anything, so it belongs in the repo beside the RLM.

Vanilla has one honest problem an RLM does not: the document may not fit. It is
truncated to `max_chars`, and the surviving fraction is REPORTED rather than
hidden -- a low vanilla score on a 3 MB document read through a 16k window is a
property of the window, not of the model, and conflating the two would flatter
the RLM enormously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class BaselineResult:
    answer: Optional[str] = None
    context_chars_sent: int = 0
    context_chars_total: int = 0
    truncated: bool = False
    end: str = "ok"

    @property
    def context_retained(self) -> float:
        if not self.context_chars_total:
            return 0.0
        return self.context_chars_sent / self.context_chars_total


def vanilla_answer(
    sub: Callable[[str, Optional[str]], str],
    document: str,
    task: str,
    *,
    max_chars: int = 60_000,
    keep: str = "head",
) -> BaselineResult:
    """One call, whole document (or as much of it as fits).

    `keep` decides WHICH part survives truncation, and it matters more than it
    looks: "head" and "tail" each score ~100% on a needle planted at that end
    and 0% at the other, so a single choice is not a fair baseline on its own.
    "middle" is the least self-serving default for a planted-needle corpus;
    sweep all three when the number is going into a table.
    """
    total = len(document)
    if total <= max_chars:
        sent = document
    elif keep == "tail":
        sent = document[-max_chars:]
    elif keep == "middle":
        mid = total // 2
        half = max_chars // 2
        sent = document[mid - half : mid + half]
    else:
        sent = document[:max_chars]

    try:
        answer = sub(task, sent)
        end = "ok"
    except Exception as exc:  # a context-overflow refusal is a real outcome
        answer, end = None, f"error: {type(exc).__name__}"

    return BaselineResult(
        answer=answer,
        context_chars_sent=len(sent),
        context_chars_total=total,
        truncated=len(sent) < total,
        end=end,
    )
