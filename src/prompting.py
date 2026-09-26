"""Canonical trace content and how each model sees it.

A trace is stored once as *content* -- {question, thinking, answer} -- and rendered per model:

    style "sgl"       the student's training text, byte-identical to SpectralGuidedLearning/data_prep.py
                      (the format the SGL / P-ALIGN / SSFT baselines were trained on): chat template with
                      enable_thinking=False, user turn "Please reason step by step, ... \\boxed{}.{problem}",
                      response "{thinking}\\n\\n\\n{answer}" (the <think> markers stripped), then the
                      tokenizer's eos token.
    style "thinking"  how a teacher reads a trace as its own reasoning: its own chat template in thinking
                      mode, response "<think>\\n{thinking}\\n</think>\\n\\n{answer}" -- without the opener
                      when the template already opens <think> (DeepSeek-R1-Distill).

Nodes (step_nodes.py) are defined on the content, so teacher and student get the same nodes even with
different tokenizers and templates; every node carries a hash of its text, which is how routing
signals computed on the teacher's rendering are matched to the student's rendering.
"""

import hashlib
import unicodedata

# SGL / P-ALIGN prompt: instruction first, no separator (SpectralGuidedLearning/src/data_prep.py).
USER_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.{question}"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
STYLES = ("sgl", "thinking")
# Generation must stop on these *token ids*: vLLM matches stop strings on detokenized text, which drops
# special tokens, and a Base model's generation_config lists only <|endoftext|> as eos.
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<｜end▁of▁sentence｜>")


def stop_token_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id}
    for token in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(i for i in ids if i is not None)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def user_content(question: str) -> str:
    return USER_TEMPLATE.format(question=nfc(question))


def render_prompt(tokenizer, question: str, style: str) -> str:
    """Chat-templated prompt ending where the assistant starts writing."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content(question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=style == "thinking",
    )


def prompt_opens_think(prompt: str) -> bool:
    return prompt.rstrip().endswith(THINK_OPEN)


def sgl_reconcile(prompt: str, response: str) -> str:
    """SpectralGuidedLearning/src/data_prep.reconcile_thinking_markers, verbatim semantics."""
    body = response.lstrip()
    if prompt.rstrip().endswith(THINK_OPEN):
        return body[len(THINK_OPEN):].lstrip("\n") if body.startswith(THINK_OPEN) else response
    return body.replace(THINK_OPEN, "", 1).replace(THINK_CLOSE, "", 1).lstrip("\n")


def render(tokenizer, content: dict, style: str) -> dict | None:
    """content {question, thinking, answer} -> {prompt, response, question_span, thinking_span, answer_span}.

    Spans are character offsets: question_span in the prompt, the others in the response. answer may
    be empty for a truncated rollout (answer_span is then None). Returns None if the rendered text
    does not contain the content verbatim (e.g. a trajectory quoting a literal "</think>").
    """
    thinking, answer = nfc(content["thinking"]).strip(), nfc(content.get("answer") or "").strip()
    prompt = render_prompt(tokenizer, content["question"], style)
    if style == "sgl":
        raw = f"{THINK_OPEN}\n{thinking}\n{THINK_CLOSE}\n\n{answer}" if answer else f"{THINK_OPEN}\n{thinking}"
        response = sgl_reconcile(prompt, nfc(raw))
    else:
        opener = "" if prompt_opens_think(prompt) else f"{THINK_OPEN}\n"
        response = f"{opener}{thinking}\n{THINK_CLOSE}\n\n{answer}" if answer else f"{opener}{thinking}"
    t0 = response.find(thinking)
    a0 = response.rfind(answer) if answer else -1
    q = user_content(content["question"])
    q0 = prompt.find(q)
    if t0 < 0 or q0 < 0 or (answer and a0 < t0 + len(thinking)):
        return None
    return {
        "prompt": prompt,
        "response": response,
        "question_span": (q0, q0 + len(q)),
        "thinking_span": (t0, t0 + len(thinking)),
        "answer_span": (a0, a0 + len(answer)) if answer else None,
    }


def split_response(text: str) -> dict:
    """{thinking, answer, closed} from a generation in either style.

    thinking style: "[<think>]...</think> answer". sgl style (no markers): "{thinking}\\n\\n\\n{answer}",
    split at the last triple newline; a generation without either separator is treated as truncated.
    """
    if THINK_CLOSE in text:
        head, answer = text.split(THINK_CLOSE, 1)
        thinking = head.split(THINK_OPEN, 1)[1] if THINK_OPEN in head else head
        return {"thinking": thinking.strip(), "answer": answer.strip(), "closed": True}
    body = text.split(THINK_OPEN, 1)[1] if THINK_OPEN in text else text
    if THINK_OPEN not in text and "\n\n\n" in body:
        thinking, answer = body.rsplit("\n\n\n", 1)
        return {"thinking": thinking.strip(), "answer": answer.strip(), "closed": True}
    return {"thinking": body.strip(), "answer": "", "closed": False}


def text_hash(text: str) -> int:
    """Stable signed 63-bit id of a node's text (first 8 bytes of SHA-1), for cross-model node matching."""
    return int.from_bytes(hashlib.sha1(nfc(text).encode("utf-8")).digest()[:8], "big") >> 1
