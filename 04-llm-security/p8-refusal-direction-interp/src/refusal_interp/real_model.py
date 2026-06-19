"""OPTIONAL enhanced path: extract a refusal direction from a real open-weight
instruct model with `transformers`.

This module imports lazily so the package still imports with ONLY the default
libs installed. It is NOT exercised by the default `make run` target or CI --
it requires downloading model weights (GPU-preferred; see README Colab note).

IMPORTANT (ethics): this code only ever computes and analyses *activations* and a
*direction vector*. It never writes out a modified set of model weights, and the
committed artifact is analysis only. See ../../ETHICS.md.
"""

from __future__ import annotations

import torch


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:  # pragma: no cover - optional path
        raise ImportError(
            "The real-model path needs `transformers` (and a downloaded model). "
            "Install with: pip install -r requirements.txt  (enhanced extras). "
            "The default offline path (make run) needs none of this."
        ) from e


def load_instruct_model(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load a small open-weight instruct model on CPU. Optional path only."""
    _require_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, output_hidden_states=True
    )
    model.eval()
    return model, tok


@torch.no_grad()
def last_token_residuals(model, tok, prompts: list[str], layer: int) -> torch.Tensor:
    """Mean-pool nothing -- collect the last-token residual at `layer` per prompt.

    Returns (N, D). This is the real analogue of synthetic.generate_activations.
    """
    feats = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt")
        out = model(**enc)
        # hidden_states: tuple(n_layers+1) of (1, seq, D); take last token at `layer`.
        h = out.hidden_states[layer][0, -1, :]
        feats.append(h)
    return torch.stack(feats, dim=0)
