from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class EmbeddingOutput:
    """Unified output from any TSFM embedding extraction."""
    # (batch, n_tokens, hidden_dim) — patch/token sequence
    embeddings: torch.Tensor
    # (batch, hidden_dim) — mean-pooled over token dim, ready for probing/regression
    pooled: torch.Tensor
    # which layer index was extracted (-1 = last)
    layer_idx: int
    # extra model-specific metadata
    meta: dict = field(default_factory=dict)


class TSFMWrapper(ABC):
    """
    Abstract base for all TSFM embedding wrappers.

    Input contract
    --------------
    x : torch.Tensor
        Shape (batch, seq_len) for univariate
        Shape (batch, seq_len, n_vars) for multivariate.
        Values should already be float32/bfloat16 — no normalisation applied here.

    Output contract
    ---------------
    EmbeddingOutput with embeddings of shape (batch, n_tokens, hidden_dim).
    """

    def __init__(self, checkpoint: str, device: str = "cpu", **kwargs):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self):
        ...

    @abstractmethod
    def get_embeddings(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        **kwargs,
    ) -> EmbeddingOutput:
        ...

    @property
    def nn_module(self) -> torch.nn.Module:
        """Return the underlying nn.Module for parameter iteration/freezing.

        Wrappers whose _load_model() returns a non-Module pipeline object
        (e.g. Chronos2Pipeline) should override this to return the inner module.
        """
        return self.model

    @property
    def num_layers(self) -> Optional[int]:
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}(checkpoint={self.checkpoint!r}, device={self.device!r})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(embeddings: torch.Tensor) -> torch.Tensor:
        """(batch, tokens, dim) → (batch, dim)"""
        return embeddings.mean(dim=1)

    @staticmethod
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        """Ensure univariate tensor is 2-D (batch, seq)."""
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x
