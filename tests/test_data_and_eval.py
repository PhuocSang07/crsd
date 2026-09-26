"""data_prep records, trace selection, eval scoring and error injection without any model download."""

import json
import re
from argparse import Namespace

import pytest

import error_injection
import generate_traces
from data_prep import build_record
from evaluate import score_file
from prompting import END_OF_TURN, user_content


class WordTokenizer:
    """Stand-in fast tokenizer: whitespace-attached words, special tokens kept whole."""

    _PIECE = re.compile(r"<\|im_(?:start|end)\|>|<think>|</think>|\s*\S+|\s+")
    unk_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(self._PIECE.finditer(text))
        out = {"input_ids": [1 + hash(m.group()) % 1000 for m in matches]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return out

    def convert_tokens_to_ids(self, token):
        return 7 if token == END_OF_TURN else self.unk_token_id


def _trace(question="What is 6 times 7?"):
    prompt = f"<|im_start|>user\n{user_content(question)}<|im_end|>\n<|im_start|>assistant\n"
    steps = [f"Step {k}: we multiply the running value by one and keep 42 in mind for later use." for k in range(12)]
    response = "<think>\n" + "\n\n".join(steps) + "\n</think>\n\nSo the answer is \\boxed{42}."
    return {"id": "t0", "question": question, "prompt": prompt, "response": response, "gold": "42"}


def _args(**overrides):
    base = dict(segment_mode="paragraph", min_step_chars=40, max_steps=400, max_tokens=32768)
    return Namespace(**{**base, **overrides})


def test_build_record_spans_and_supervision():
    record, reason = build_record(WordTokenizer(), _trace(), _args())
    assert reason == "ok"
    assert [n["kind"] for n in record["nodes"]] == ["question"] + ["step"] * 12 + ["answer"]
    assert record["input_ids"][-1] == 7  # <|im_end|> appended as the stop token
    start, end = record["response_token_span"]
    assert end == len(record["input_ids"]) - 1
    for node in record["nodes"]:
        assert node["token_end"] > node["token_start"]
    assert record["nodes"][1]["token_start"] >= start  # steps live in the response
    assert record["nodes"][0]["token_end"] <= start
    assert build_record(WordTokenizer(), _trace(), _args(max_tokens=20))[1] == "too_long"


def test_reference_answer_and_selection(tmp_path):
    assert generate_traces.reference_answer("so \\boxed{12}") == "12"
    assert generate_traces.reference_answer("128") == "128"
    assert generate_traces.reference_answer("1. **Proof**\nlong text") is None
    row = {**_trace(), "reference": "42", "source": "x", "cot_type": "math", "gold": "42", "generations": [
        {"text": "<think>\nwork\n</think>\n\n\\boxed{42}", "finish_reason": "stop", "n_tokens": 10},
        {"text": "<think>\nwork\n</think>\n\n\\boxed{41}", "finish_reason": "stop", "n_tokens": 10},
        {"text": "<think>\nnever closes", "finish_reason": "length", "n_tokens": 99},
        {"text": "<think>\nagain\n</think>\n\nIt is \\boxed{42}.", "finish_reason": "stop", "n_tokens": 12},
    ]}
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(row) + "\n")
    out = tmp_path / "traces.jsonl"
    generate_traces.select(Namespace(raw_path=str(raw), judge_model=None, seed=0, output_path=str(out), max_keep=None,
                                     tensor_parallel_size=1))
    kept = [json.loads(line) for line in out.open()]
    assert len(kept) == 1 and kept[0]["teacher_solve_rate"] == 0.5 and len(kept[0]["candidates"]) == 3
    assert sorted(str(c["correct"]) for c in kept[0]["candidates"]) == ["False", "None", "True"]
    stats = json.loads((tmp_path / "traces.jsonl.stats.json").read_text())
    assert stats["truncated_or_unclosed"] == 1 and stats["correct"] == 2


def test_score_file_unbiased_pass_at_k(tmp_path):
    raw = tmp_path / "aime24.jsonl"
    rows = [
        {"id": 0, "gold": "5", "task_type": "math", "generations": [{"text": "\\boxed{5}"}, {"text": "\\boxed{4}"}, {"text": "\\boxed{5}"}, {"text": "no"}]},
        {"id": 1, "gold": "9", "task_type": "math", "generations": [{"text": "\\boxed{1}"}] * 4},
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary, labels = score_file(raw)
    assert labels == [[1, 0, 1, 0], [0, 0, 0, 0]]
    assert summary["pass@1"] == pytest.approx(0.25)
    assert summary["pass@3"] == pytest.approx(0.5 * 1.0)  # 1 - C(2,3)/C(4,3) = 1 for problem 0
    assert summary["no_answer_rate"] == pytest.approx(1 / 8)


def _injection_record(steps):
    trace = _trace()
    trace["response"] = "<think>\n" + "\n\n".join(steps) + "\n</think>\n\nThus \\boxed{42}."
    record, _ = build_record(WordTokenizer(), trace, _args())
    return record


def test_error_injection_build_and_score():
    filler = [f"Filler reasoning step number {k} that does nothing important at all." for k in range(20)]
    record = _injection_record(["We set the base value to 137 and write it down for later.", *filler,
                                "Recall the base value 137 from before and use it again now."])
    cases = error_injection.build([record], per_distance=5, seed=0)
    injected = [c for c in cases if not c["control"]]
    assert len(injected) == 1 and injected[0]["distance"] == 16 and injected[0]["first_reuse_gap"] == 21
    case = injected[0]
    assert "137" not in error_injection.numbers(case["prefix"]) and case["wrong"] != "137"
    assert "137" in error_injection.numbers([c for c in cases if c["control"]][0]["prefix"])
    for c in cases:
        c["continuations"] = (["Wait, recompute: the base value is 137. \\boxed{42}", "Continue. \\boxed{40}"]
                              if not c["control"] else ["Using 137 again. \\boxed{42}", "Using 137. \\boxed{42}"])
    report = error_injection.score(cases)
    assert report["injected/16"]["detection"] == pytest.approx(0.5)
    assert report["injected/16"]["recovery"] == pytest.approx(0.5)
    assert report["control/16"]["states_r"] == pytest.approx(1.0)
    assert report["injected/all"]["detection_net"] == pytest.approx(-0.5)


def test_error_injection_skips_values_visible_in_prefix():
    filler = [f"Filler reasoning step number {k} that does nothing important at all." for k in range(20)]
    record = _injection_record(["We set the base value to 137 and write it down for later.",
                                "Keep 137 in mind while we do something else entirely here.", *filler,
                                "Recall the base value 137 from before and use it again now."])
    assert error_injection.build([record], per_distance=5, seed=0) == []
    assert error_injection.distance_bucket(3) is None and error_injection.distance_bucket(64) == 64
