"""Step nodes on character spans (proposal Sec. 4.1, Appendix B).

A sample's node set is V = {v0 = q, v1..vn = steps of the thinking part, v_{n+1} = a}; every node
is a half-open character span [beta, eps) of the full text prompt + response. Because nodes live in
character space, teacher and student always get the same n nodes regardless of tokenizer; each
model maps them onto its own tokens via offset mapping:

    I_M(v_k) = {t : first character of token t in [beta_k, eps_k)}

Default segmentation (A9 "paragraph"): split the thinking text at "\\n\\n" (the separator stays
with the step before it, so reflection words like "Wait" open a step), merge pieces shorter than
40 characters into the previous one, and while n > 400 merge the adjacent pair with the smallest
combined length. The other A9 modes are "sentence", "episode" (a new step only at a paragraph
opening with a reflection keyword) and "chunk3" (three default steps per node).
"""

import bisect
import re

from prompting import THINK_CLOSE, THINK_OPEN

MIN_STEP_CHARS = 40
MAX_STEPS = 400
SEGMENT_MODES = ("paragraph", "sentence", "episode", "chunk3")

_PARAGRAPH_BREAK = re.compile(r"\n\n+")
_SENTENCE_BREAK = re.compile(r"([.?!\}\]])([\s\n]+)([A-Z])")
# Episode openers, after Wang et al. (2026)'s keyword-delimited segments.
_EPISODE_OPENER = re.compile(
    r"^\s*(wait|alternatively|hmm|but wait|let me (?:check|verify|double|re)|actually|"
    r"hold on|so,? the answer|therefore|another (?:way|approach)|now,? let)",
    re.IGNORECASE,
)


def _paragraph_cuts(text: str) -> list[int]:
    """Start offsets of every piece: after each run of blank lines, keeping the run in front."""
    return [0] + [m.end() for m in _PARAGRAPH_BREAK.finditer(text) if m.end() < len(text)]


def _sentence_cuts(text: str) -> list[int]:
    return [0] + [m.start(3) for m in _SENTENCE_BREAK.finditer(text)]


def _spans_from_cuts(cuts: list[int], length: int) -> list[list[int]]:
    cuts = sorted(set(cuts))
    return [[start, end] for start, end in zip(cuts, cuts[1:] + [length]) if end > start]


def _merge_short(spans: list[list[int]], min_chars: int) -> list[list[int]]:
    """Fold every piece shorter than min_chars into its predecessor (the first into its successor)."""
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and end - start < min_chars:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_chars:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return merged


def _cap_steps(spans: list[list[int]], max_steps: int) -> list[list[int]]:
    """Merge the adjacent pair with the smallest combined length until at most max_steps remain."""
    spans = [list(span) for span in spans]
    while len(spans) > max_steps:
        lengths = [spans[k + 1][1] - spans[k][0] for k in range(len(spans) - 1)]
        k = min(range(len(lengths)), key=lengths.__getitem__)
        spans[k] = [spans[k][0], spans[k + 1][1]]
        del spans[k + 1]
    return spans


def split_steps(
    thinking: str,
    mode: str = "paragraph",
    min_chars: int = MIN_STEP_CHARS,
    max_steps: int = MAX_STEPS,
) -> list[tuple[int, int]]:
    """Character spans (relative to `thinking`) of its steps; together they cover it exactly."""
    if mode not in SEGMENT_MODES:
        raise ValueError(f"unknown segmentation mode {mode!r}; expected one of {SEGMENT_MODES}")
    if not thinking:
        return []
    if mode == "sentence":
        spans = _spans_from_cuts(_sentence_cuts(thinking), len(thinking))
    else:
        spans = _spans_from_cuts(_paragraph_cuts(thinking), len(thinking))
        if mode == "episode":
            spans = _spans_from_cuts(
                [start for start, end in spans if start == 0 or _EPISODE_OPENER.match(thinking[start:end])],
                len(thinking),
            )
    spans = _merge_short(spans, min_chars)
    if mode == "chunk3":
        spans = [[spans[k][0], spans[min(k + 2, len(spans) - 1)][1]] for k in range(0, len(spans), 3)]
    spans = _cap_steps(spans, max_steps)
    return [(start, end) for start, end in spans]


def build_nodes(
    prompt: str,
    response: str,
    question_span: tuple[int, int],
    mode: str = "paragraph",
    min_chars: int = MIN_STEP_CHARS,
    max_steps: int = MAX_STEPS,
    allow_unclosed: bool = False,
) -> list[dict] | None:
    """Nodes [q, s1..sn, a] as absolute character spans in prompt + response.

    The thinking part is the text strictly between <think> and </think>; a is the text after
    </think> with surrounding whitespace excluded. The markers themselves belong to no node.
    Returns None for a response without a closed thinking block or with an empty answer --
    exactly the truncated traces the filter drops anyway -- unless allow_unclosed: then a
    truncated response (no </think>) yields [q, s1..sn] with no answer node. D3 needs this, since
    dropping truncated student rollouts (nearly all wrong) would bias the error-prediction test.
    """
    open_at = response.find(THINK_OPEN)
    close_at = response.find(THINK_CLOSE)
    if open_at < 0:
        return None
    if close_at < open_at:
        if not allow_unclosed:
            return None
        close_at = len(response)
    base = len(prompt)
    think_start = open_at + len(THINK_OPEN)
    thinking = response[think_start:close_at]
    # Leading/trailing whitespace of the thinking block (the "\n" after <think>) stays node-less.
    lead = len(thinking) - len(thinking.lstrip())
    body = thinking.strip()

    unclosed = close_at == len(response) and not response.endswith(THINK_CLOSE)
    answer_raw = "" if unclosed else response[close_at + len(THINK_CLOSE) :]
    answer_lead = len(answer_raw) - len(answer_raw.lstrip())
    answer = answer_raw.strip()
    if not body or (not answer and not unclosed):
        return None

    nodes = [{"kind": "question", "char_start": question_span[0], "char_end": question_span[1]}]
    offset = base + think_start + lead
    for start, end in split_steps(body, mode=mode, min_chars=min_chars, max_steps=max_steps):
        nodes.append({"kind": "step", "char_start": offset + start, "char_end": offset + end})
    if not unclosed:
        answer_start = base + close_at + len(THINK_CLOSE) + answer_lead
        nodes.append({"kind": "answer", "char_start": answer_start, "char_end": answer_start + len(answer)})
    return nodes


def assign_token_spans(nodes: list[dict], token_starts: list[int]) -> list[tuple[int, int]] | None:
    """Token range [start, end) of every node: tokens whose first character falls in its span.

    token_starts must be non-decreasing (one tokenizer pass per segment, shifted to absolute
    character positions). Returns None if any node would own no token.
    """
    spans = []
    for node in nodes:
        start = bisect.bisect_left(token_starts, node["char_start"])
        end = bisect.bisect_left(token_starts, node["char_end"])
        if end <= start:
            return None
        spans.append((start, end))
    return spans


def token_node_ids(token_spans: list[tuple[int, int]], length: int) -> list[int]:
    """Per-token node index, -1 for tokens in no node (template, <think> markers, stop token)."""
    ids = [-1] * length
    for index, (start, end) in enumerate(token_spans):
        for t in range(start, end):
            ids[t] = index
    return ids
