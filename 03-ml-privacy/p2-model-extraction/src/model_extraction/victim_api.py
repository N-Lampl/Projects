"""The victim exposed as a black-box prediction *API*.

This is the only surface the thief is allowed to touch: send images in, get
predicted labels back. It is the natural place to put a defense, so the API
implements a QUERY BUDGET / RATE LIMIT.

`VictimAPI.predict(x)` returns hard labels (argmax) -- the most restrictive,
information-poor response an API can give, which is exactly the realistic
setting for label-only model stealing (Tramer et al., 2016).

Every call is metered. Once `max_queries` is exhausted the API raises
`QueryBudgetExceeded`, modelling a defender that rate-limits / cuts off a client
that has spent its quota. `queries_used` lets experiments measure the attacker's
realised budget.
"""

from __future__ import annotations

import torch
from torch import nn


class QueryBudgetExceeded(RuntimeError):
    """Raised when a client exceeds the API's query budget (the defense)."""


class VictimAPI:
    def __init__(
        self,
        model: nn.Module,
        *,
        max_queries: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """`max_queries=None` means unlimited (no defense); an int enables the
        rate-limit defense by capping total images served across all calls."""
        self.model = model.eval()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.max_queries = max_queries
        self.queries_used = 0

    @property
    def budget_remaining(self) -> int | None:
        if self.max_queries is None:
            return None
        return max(self.max_queries - self.queries_used, 0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return hard predicted labels for a batch of images.

        Counts `len(x)` against the budget. If the batch would exceed the budget
        the WHOLE call is rejected (an honest API would not partially serve)."""
        n = x.shape[0]
        if self.max_queries is not None and self.queries_used + n > self.max_queries:
            raise QueryBudgetExceeded(
                f"request for {n} predictions would exceed budget "
                f"({self.queries_used}/{self.max_queries} used)"
            )
        self.queries_used += n
        logits = self.model(x.to(self.device))
        return logits.argmax(1).cpu()
