"""Step nodes on character spans (Sec. 4.1, Appendix B)."""

from prompting import render, split_response, user_content
from step_nodes import assign_token_spans, build_nodes, split_steps
from stubs import R1Tokenizer, WordTokenizer, content


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


def test_render_sgl_matches_the_baselines_format():
    tok, c = WordTokenizer(), content(3)
    r = render(tok, c, "sgl")
    assert r["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert user_content(c["question"]) in r["prompt"] and user_content("x").startswith("Please reason step by step")
    assert r["response"] == c["thinking"] + "\n\n\n" + c["answer"]  # SGL strips the markers
    t0, t1 = r["thinking_span"]
    a0, a1 = r["answer_span"]
    assert r["response"][t0:t1] == c["thinking"] and r["response"][a0:a1] == c["answer"]


def test_render_thinking_for_qwen3_and_r1_templates():
    c = content(3)
    qwen = render(WordTokenizer(), c, "thinking")
    assert qwen["response"] == f"<think>\n{c['thinking']}\n</think>\n\n{c['answer']}"
    r1 = render(R1Tokenizer(), c, "thinking")
    assert r1["prompt"].endswith("<think>\n") and r1["response"].startswith(c["thinking"][:10])
    assert "<think>" not in r1["response"]


def test_node_hashes_agree_across_styles_and_tokenizers():
    c = content(12)
    views = [render(WordTokenizer(), c, "sgl"), render(WordTokenizer(), c, "thinking"), render(R1Tokenizer(), c, "thinking")]
    node_sets = [build_nodes(v) for v in views]
    hashes = [[n["hash"] for n in nodes] for nodes in node_sets]
    assert hashes[0] == hashes[1] == hashes[2]
    assert [n["kind"] for n in node_sets[0]] == ["question"] + ["step"] * 12 + ["answer"]
    for view, nodes in zip(views, node_sets):
        text = view["prompt"] + view["response"]
        assert text[nodes[0]["char_start"]:nodes[0]["char_end"]] == user_content(c["question"])
        assert text[nodes[-1]["char_start"]:nodes[-1]["char_end"]] == c["answer"]
        assert "<think>" not in "".join(text[n["char_start"]:n["char_end"]] for n in nodes)


def test_truncated_content_has_no_answer_node():
    c = {**content(5), "answer": ""}
    nodes = build_nodes(render(WordTokenizer(), c, "sgl"))
    assert nodes[-1]["kind"] == "step" and len(nodes) == 6


def test_split_response_both_styles():
    assert split_response("<think>\nwork\n</think>\n\nans") == {"thinking": "work", "answer": "ans", "closed": True}
    assert split_response("work\n</think>\n\nans")["thinking"] == "work"  # R1 template opened <think>
    assert split_response("step one\n\nstep two\n\n\nFinal \\boxed{1}")["answer"] == "Final \\boxed{1}"
    assert split_response("<think>\nnever closes")["closed"] is False
    assert split_response("no separator at all")["closed"] is False


def test_assign_token_spans_with_merged_tokens():
    nodes = [{"char_start": 0, "char_end": 5}, {"char_start": 5, "char_end": 12}]
    # a token starting at 4 belongs to node 0 even though it straddles the boundary
    assert assign_token_spans(nodes, [0, 2, 4, 7, 9]) == [(0, 3), (3, 5)]
    assert assign_token_spans(nodes, [0, 6]) is not None
    assert assign_token_spans([{"char_start": 0, "char_end": 1}, {"char_start": 1, "char_end": 2}], [0, 5]) is None
