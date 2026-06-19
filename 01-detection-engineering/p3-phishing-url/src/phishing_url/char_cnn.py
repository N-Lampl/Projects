"""OPTIONAL char-level CNN over raw URL strings (torch).

This is the deep-learning alternative to the lexical+sklearn default. torch is
imported lazily inside the functions so this module imports fine without it; the
sklearn path stays the project's offline default. CPU-friendly: tiny embedding +
one conv, a few epochs on a few thousand short strings.
"""

from __future__ import annotations

import numpy as np

# printable ASCII vocab; index 0 is reserved for padding/unknown
_VOCAB = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`" \
         "abcdefghijklmnopqrstuvwxyz{|}~"
_CHAR2IDX = {c: i + 1 for i, c in enumerate(_VOCAB)}
VOCAB_SIZE = len(_VOCAB) + 1
MAX_LEN = 200


def encode_urls(urls, max_len: int = MAX_LEN) -> np.ndarray:
    """Map URL strings to fixed-length integer index arrays (right-padded)."""
    out = np.zeros((len(urls), max_len), dtype=np.int64)
    for i, u in enumerate(urls):
        for j, ch in enumerate(u[:max_len]):
            out[i, j] = _CHAR2IDX.get(ch, 0)
    return out


def build_cnn(embed_dim: int = 16, n_filters: int = 32):
    """Return an unfitted CharCNN (torch.nn.Module). Imports torch lazily."""
    import torch.nn as nn

    class CharCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
            self.conv = nn.Conv1d(embed_dim, n_filters, kernel_size=5, padding=2)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(n_filters, 2)

        def forward(self, x):  # x: (B, L) long
            e = self.embed(x).transpose(1, 2)  # (B, embed, L)
            h = self.act(self.conv(e))
            h = self.pool(h).squeeze(-1)  # (B, n_filters)
            return self.fc(h)

    return CharCNN()


def train_cnn(model, X: np.ndarray, y: np.ndarray, epochs: int = 4, batch_size: int = 128,
              lr: float = 1e-3, device: str = "cpu") -> None:
    """Train the CharCNN in place. Imports torch lazily."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to(device).train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
    model.eval()


def predict_proba_cnn(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return P(phishing) for each row. Imports torch lazily."""
    import torch

    model.to(device).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
