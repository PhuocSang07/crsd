"""Answer extraction and equivalence checking for math and multiple-choice benchmarks.

Math answers are compared symbolically via math-verify when available, with a normalized
string fallback so scoring still runs in environments without it.
"""

import re

_BOXED = re.compile(r"\\boxed\s*{")
_MCQ_PATTERNS = [
    re.compile(r"\\boxed\s*{\s*\(?([A-D])\)?\s*}"),
    re.compile(r"(?:answer|Answer)\s*(?:is)?\s*[:\-]?\s*\(?([A-D])\)?\b"),
    re.compile(r"\b([A-D])\s*$"),
]


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with brace matching for nested expressions."""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None

    start = matches[-1].end()
    depth, index = 1, start
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    return text[start : index - 1].strip() if depth == 0 else text[start:].strip()


def normalize_math(answer: str) -> str:
    """Lowercase, strip formatting noise that never changes mathematical meaning."""
    answer = answer.strip().rstrip(".").replace(" ", "")
    answer = re.sub(r"\\(?:left|right|text|mathrm|displaystyle)", "", answer)
    answer = re.sub(r"\\[!,;:]", "", answer)  # LaTeX spacing commands
    answer = re.sub(r"\\%|%|\\\$|\$", "", answer)
    answer = re.sub(r"^\{(.*)\}$", r"\1", answer)
    answer = re.sub(r"\.0+$", "", answer)
    return answer.lower()


def math_answers_equal(predicted: str | None, gold: str) -> bool:
    """Symbolic comparison when math-verify is installed, normalized-string otherwise."""
    if predicted is None:
        return False
    if normalize_math(predicted) == normalize_math(gold):
        return True

    try:
        from math_verify import parse, verify
    except ImportError:
        return False

    try:
        return bool(verify(parse(f"${gold}$"), parse(f"${predicted}$")))
    except Exception:
        return False


def extract_choice(text: str) -> str | None:
    """First multiple-choice letter found by the ordered patterns, searching the tail first."""
    tail = text[-500:] if len(text) > 500 else text
    for pattern in _MCQ_PATTERNS:
        match = pattern.search(tail) or pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def score_generation(generation: str, gold: str, task_type: str) -> bool:
    """True if the generation's final answer matches gold. task_type: 'math' | 'mcq'."""
    if task_type == "mcq":
        return extract_choice(generation) == gold.strip().upper()
    return math_answers_equal(extract_boxed(generation), gold)
