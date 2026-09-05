"""rlmadp -- a minimal Recursive Language Model, written to be read."""

from .rlm import RLM, RLMResult, Turn, final_literals, static_string

__all__ = ["RLM", "RLMResult", "Turn", "final_literals", "static_string"]
__version__ = "0.1.0"
