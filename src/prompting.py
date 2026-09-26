"""One prompt format for trace generation, training, teacher-forcing and evaluation.

The teacher (Qwen3-8B, thinking mode) writes the traces from this prompt and the student is trained
on exactly the same text, so teacher and student read byte-identical sequences and their step
nodes line up one-to-one (proposal Sec. 5: every diagnostic is teacher-forced on one text).

The prompt is rendered once with the *teacher's* chat template and stored verbatim in the trace
file; Qwen3-*-Base tokenizers carry the same special tokens, so no second rendering is needed.
"""

# Qwen3's recommended math instruction (model card), question first.
MATH_INSTRUCTION = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# End-of-turn token appended to every training target. Qwen3-Base's eos is <|endoftext|>, but a
# chat-formatted student has to learn to close the assistant turn, and vLLM stops on it at eval.
END_OF_TURN = "<|im_end|>"
# Generation must stop on these *token ids*: vLLM matches stop strings on detokenized text, which drops
# special tokens, so a "<|im_end|>" stop string never fires -- and a Base model's generation_config only
# lists <|endoftext|> as eos.
STOP_TOKENS = (END_OF_TURN, "<|endoftext|>")


def stop_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    return [i for i in ids if i is not None and i != tokenizer.unk_token_id]


def user_content(question: str) -> str:
    return MATH_INSTRUCTION.format(question=question.strip())


def render_prompt(tokenizer, question: str, enable_thinking: bool = True) -> str:
    """Chat-templated generation prompt, ending right where the assistant starts writing."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content(question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def question_char_span(prompt: str, question: str) -> tuple[int, int]:
    """Character span of node v0 = q inside the rendered prompt.

    v0 covers the whole user content (instruction + question; Qwen3 has no default system
    prompt). Template tokens (<|im_start|>user ...) belong to no node, which also keeps the
    attention-sink first token out of every routing distribution.
    """
    content = user_content(question)
    start = prompt.find(content)
    if start < 0:
        raise ValueError("user content not found verbatim in the rendered prompt")
    return start, start + len(content)
