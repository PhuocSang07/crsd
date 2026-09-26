"""P-ALIGN's answer grader, ported from P-ALIGN/src/evaluation.py.

A generation is correct when EITHER math_verify OR oat_math_grader accepts it (their
`any_true=True` default). oat_math_grader is imported by their evaluation.py but missing
from their requirements.txt, so `grade()` reports which branches are live instead of
silently degrading.
"""

import os
import signal
from contextlib import contextmanager

VERIFY_TIMEOUT_SECONDS = 10


@contextmanager
def _time_limit(seconds: int):
    """Abort a hanging sympy verification. POSIX-only, as in P-ALIGN's own decorator."""
    if os.name != "posix":
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("verification timed out")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _math_verify_labels(predictions: list[str], gold: str) -> list[int]:
    from math_verify import parse, verify

    try:
        with _time_limit(VERIFY_TIMEOUT_SECONDS):
            parsed_gold = parse("$" + gold + "$")  # P-ALIGN wraps gold in $..$ before parsing
            return [int(bool(verify(parsed_gold, parse(prediction)))) for prediction in predictions]
    except Exception:
        return [0] * len(predictions)


def _oat_label(prediction: str, gold: str) -> int:
    from oat_math_grader import boxed_reward_fn

    try:
        _, result = boxed_reward_fn(prediction, gold, fast=False)
        return int(result == 1.0)
    except Exception:
        return 0


def active_graders() -> list[str]:
    """Grader branches whose imports resolve in this environment."""
    available = []
    for module in ("math_verify", "oat_math_grader"):
        try:
            __import__(module)
        except ImportError:
            continue
        available.append(module)
    return available


def grade(predictions: list[str], gold: str) -> list[int]:
    """Per-generation 0/1 labels for one problem. Raises if no grader is installed."""
    available = active_graders()
    if not available:
        raise ImportError(
            "P-ALIGN grading needs math_verify (and optionally oat_math_grader); install "
            "math-verify, or pass --grader builtin to use this repo's own scorer."
        )

    labels = (
        _math_verify_labels(predictions, gold)
        if "math_verify" in available
        else [0] * len(predictions)
    )
    if "oat_math_grader" in available:
        labels = [
            int(label or _oat_label(prediction, gold))
            for label, prediction in zip(labels, predictions)
        ]
    return labels
