"""rlmadp -- a minimal Recursive Language Model, written to be read.

Implements the paradigm of Zhang, Kraska & Khattab, "Recursive Language
Models" (arXiv:2512.24601): the prompt is not context, it is an external
environment -- a variable in a Python REPL that the model programmatically
examines, decomposes, and recursively calls itself over.
"""

from .baseline import BaselineResult, vanilla_answer
from .rlm import RLM, RLMResult, Turn, final_literals, static_string

__all__ = [
    "RLM", "RLMResult", "Turn", "final_literals", "static_string",
    "vanilla_answer", "BaselineResult",
]
__version__ = "0.2.0"
