"""Tokenizer stand-ins with chat templates, so rendering/node tests need no model download."""

import re


class WordTokenizer:
    """Whitespace-attached words, special tokens whole; Qwen3-like chat template."""

    _PIECE = re.compile(r"<\|im_(?:start|end)\|>|<｜[^｜]+｜>|<think>|</think>|\s*\S+|\s+")
    unk_token_id = 0
    eos_token_id = 7

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(self._PIECE.finditer(text))
        out = {"input_ids": [11 + hash(m.group()) % 1000 for m in matches]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(m.start(), m.end()) for m in matches]
        return out

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 7, "<|endoftext|>": 8}.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        prompt = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        return prompt if enable_thinking else prompt + "<think>\n\n</think>\n\n"


class R1Tokenizer(WordTokenizer):
    """DeepSeek-R1-Distill-like template: opens <think> itself, ignores enable_thinking."""

    eos_token_id = 9

    def convert_tokens_to_ids(self, token):
        return {"<｜end▁of▁sentence｜>": 9}.get(token, self.unk_token_id)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return f"<｜begin▁of▁sentence｜><｜User｜>{messages[0]['content']}<｜Assistant｜><think>\n"


def content(n_steps: int = 12, question: str = "What is 6 times 7?", answer: str = "So the answer is \\boxed{42}.") -> dict:
    steps = [f"Step {k}: we multiply the running value by one and keep 42 in mind for later use." for k in range(n_steps)]
    return {"id": "t0", "question": question, "thinking": "\n\n".join(steps), "answer": answer, "gold": "42", "closed": True}
