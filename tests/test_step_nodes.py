"""Step nodes on character spans (Sec. 4.1, Appendix B)."""

from prompting import question_char_span, user_content
from step_nodes import assign_token_spans, build_nodes, split_steps, token_node_ids


def _covers(text, spans):
    return "".join(text[s:e] for s, e in spans) == text and all(e > s for s, e in spans)


def test_paragraph_split_covers_text_and_keeps_separator_with_previous_step():
    text = "First paragraph that is long enough to stand alone.\n\nWait, second paragraph also long enough here.\n\nThird paragraph, long enough as well for sure."
    spans = split_steps(text)
    assert _covers(text, spans) and len(spans) == 3
    assert text[spans[1][0]:].startswith("Wait")
    assert text[spans[0][0]:spans[0][1]].endswith("\n\n")


def test_short_pieces_merge_into_previous():
    text = "A paragraph that is definitely longer than forty characters.\n\nOk.\n\nAnother paragraph that is definitely longer than forty chars."
    spans = split_steps(text)
    assert _covers(text, spans) and len(spans) == 2
    assert "Ok." in text[spans[0][0]:spans[0][1]]


def test_first_short_piece_merges_forward():
    text = "Hi.\n\nA paragraph that is definitely longer than forty characters."
    spans = split_steps(text)
    assert spans == [(0, len(text))]


def test_cap_merges_shortest_adjacent_pairs():
    paragraphs = [("x" * (41 + (k % 7))) for k in range(50)]
    text = "\n\n".join(paragraphs)
    spans = split_steps(text, max_steps=20)
    assert len(spans) == 20 and _covers(text, spans)


def test_other_modes_cover_text():
    text = "\n\n".join([
        "Let us set up the problem carefully with all the variables.",
        "Compute the first quantity to be twelve point five exactly.",
        "Wait, that seems off, so let me reconsider the first value.",
        "Now the second quantity equals seven. Then we add them up.",
        "Hmm, maybe there is a cleaner way to see the whole thing.",
    ])
    for mode in ("sentence", "episode", "chunk3"):
        spans = split_steps(text, mode=mode)
        assert _covers(text, spans), mode
    assert len(split_steps(text, mode="chunk3")) == 2
    episode = split_steps(text, mode="episode")
    assert [text[s:].split()[0] for s, _ in episode] == ["Let", "Wait,", "Hmm,"]


def test_build_nodes_and_token_mapping():
    question = "What is 2+2?"
    prompt = f"<|im_start|>user\n{user_content(question)}<|im_end|>\n<|im_start|>assistant\n"
    steps = ["We need to add two and two, which is a simple sum.", "Two plus two gives four, so the result is four."]
    response = "<think>\n" + "\n\n".join(steps) + "\n</think>\n\nThe answer is \\boxed{4}."
    nodes = build_nodes(prompt, response, question_char_span(prompt, question))
    text = prompt + response
    assert [n["kind"] for n in nodes] == ["question", "step", "step", "answer"]
    assert text[nodes[0]["char_start"]:nodes[0]["char_end"]] == user_content(question)
    assert text[nodes[1]["char_start"]:nodes[1]["char_end"]].startswith("We need")
    assert text[nodes[2]["char_start"]:nodes[2]["char_end"]] == steps[1]
    assert text[nodes[3]["char_start"]:nodes[3]["char_end"]] == "The answer is \\boxed{4}."
    # character-level "tokenizer": token t starts at character t
    spans = assign_token_spans(nodes, list(range(len(text))))
    assert spans[0] == (nodes[0]["char_start"], nodes[0]["char_end"])
    ids = token_node_ids(spans, len(text))
    assert ids[text.index("<think>")] == -1 and ids[text.index("</think>")] == -1 and ids[0] == -1


def test_build_nodes_rejects_unclosed_or_empty_answer():
    q = "q"
    prompt = user_content(q)
    span = question_char_span(prompt, q)
    assert build_nodes(prompt, "<think>\nlong reasoning without end", span) is None
    assert build_nodes(prompt, "<think>\nsome reasoning here\n</think>\n\n", span) is None


def test_assign_token_spans_with_merged_tokens():
    nodes = [{"char_start": 0, "char_end": 5}, {"char_start": 5, "char_end": 12}]
    # a token starting at 4 belongs to node 0 even though it straddles the boundary
    assert assign_token_spans(nodes, [0, 2, 4, 7, 9]) == [(0, 3), (3, 5)]
    assert assign_token_spans(nodes, [0, 6]) is not None
    assert assign_token_spans([{"char_start": 0, "char_end": 1}, {"char_start": 1, "char_end": 2}], [0, 5]) is None
