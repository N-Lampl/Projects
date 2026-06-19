"""Optional char-level LSTM detector (enhanced path).

torch IS in the default deps, but this module keeps the import lazy and guarded so
the package still imports cleanly if torch is unavailable. The default
``make detect`` uses the scikit-learn model; pass ``--lstm`` to train this one.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from .features import second_level

# index 0 is reserved for padding / unknown chars
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-_."
_VOCAB = {c: i + 1 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(_VOCAB) + 1
MAX_LEN = 40


def encode(domains: list[str], max_len: int = MAX_LEN) -> "np.ndarray":
    """Map each domain label to a fixed-length integer sequence."""
    out = np.zeros((len(domains), max_len), dtype=np.int64)
    for i, d in enumerate(domains):
        s = second_level(d)[:max_len]
        for j, ch in enumerate(s):
            out[i, j] = _VOCAB.get(ch, 0)
    return out


if _HAS_TORCH:

    class CharLSTM(nn.Module):
        """Embedding -> single-layer LSTM -> linear head (2 classes)."""

        def __init__(self, embed_dim: int = 24, hidden: int = 32):
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
            self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            e = self.embed(x)
            _, (h, _) = self.lstm(e)
            return self.head(h[-1])

    def train_lstm(
        domains: list[str],
        y: np.ndarray,
        epochs: int = 4,
        batch_size: int = 128,
        lr: float = 1e-2,
    ) -> "CharLSTM":
        """Train the char-LSTM on CPU. Small enough to finish in seconds."""
        x = torch.from_numpy(encode(domains))
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
        model = CharLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            model.train()
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(x[idx]), yt[idx])
                loss.backward()
                opt.step()
        return model.eval()

    @torch.no_grad()
    def lstm_proba(model: "CharLSTM", domains: list[str]) -> "np.ndarray":
        logits = model(torch.from_numpy(encode(domains)))
        return torch.softmax(logits, dim=1)[:, 1].numpy()

else:  # pragma: no cover - torch missing

    class CharLSTM:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("torch is required for the char-LSTM path")

    def train_lstm(*a, **k):
        raise ImportError("torch is required for the char-LSTM path")

    def lstm_proba(*a, **k):
        raise ImportError("torch is required for the char-LSTM path")
