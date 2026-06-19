"""Generation + measurement helpers for the real-model path.

We deliberately generate only a SHORT prefix and classify refuse-vs-comply from
it. The model's full completions for harmful prompts are never stored or returned
to the caller beyond what's needed to detect a refusal marker. See ../../ETHICS.md.
"""

from __future__ import annotations

import torch

from .prompts import REFUSAL_MARKERS


def is_refusal(prefix: str) -> bool:
    """True if a generated prefix reads as a refusal."""
    low = prefix.strip().lower()
    if not low:
        return True  # empty / immediate stop counts as non-compliance == refusal
    return any(m in low[:160] for m in REFUSAL_MARKERS)


@torch.no_grad()
def generate_prefix(model, tok, prompt: str, max_new_tokens: int = 24) -> str:
    """Greedy-decode a short prefix for `prompt`. Returns only the NEW text."""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt")
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def refusal_rate_real(model, tok, prompts: list[str], max_new_tokens: int = 24) -> float:
    """Fraction of `prompts` the model refuses. Completions are not retained."""
    refusals = 0
    for p in prompts:
        if is_refusal(generate_prefix(model, tok, p, max_new_tokens)):
            refusals += 1
    return refusals / max(len(prompts), 1)


@torch.no_grad()
def perplexity(model, tok, text: str) -> float:
    """LM perplexity on a fixed benign passage (capability-retention proxy)."""
    enc = tok(text, return_tensors="pt")
    out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())
