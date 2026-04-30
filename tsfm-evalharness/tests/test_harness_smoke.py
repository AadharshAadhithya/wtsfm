"""
Smoke tests for the harness — run without any model downloads.
These test the registry wiring, base class, and lazy-loading mechanics
using a minimal stub model so nothing is fetched from the internet.
"""
import torch
import pytest

from harness.base import TSFMWrapper, EmbeddingOutput
from harness.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Stub wrapper — avoids real downloads
# ---------------------------------------------------------------------------

class StubWrapper(TSFMWrapper):
    """Returns random embeddings of shape (B, 8, 64)."""

    def _load_model(self):
        return object()  # dummy

    def get_embeddings(self, x, layer_idx=-1, **kwargs):
        x = self._to_2d(x)
        B = x.shape[0]
        emb = torch.randn(B, 8, 64)
        return EmbeddingOutput(
            embeddings=emb,
            pooled=self._mean_pool(emb),
            layer_idx=layer_idx,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_available_models():
    models = ModelRegistry.available()
    for expected in ("patchtst", "chronos2", "timesfm", "sundial", "moirai2", "tirex", "toto"):
        assert expected in models, f"{expected} missing from registry"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelRegistry.load("does_not_exist")


def test_stub_univariate():
    w = StubWrapper(checkpoint="stub", device="cpu")
    x = torch.randn(4, 512)
    out = w.get_embeddings(x)
    assert isinstance(out, EmbeddingOutput)
    assert out.embeddings.shape == (4, 8, 64)
    assert out.pooled.shape == (4, 64)
    assert out.layer_idx == -1


def test_stub_single_sample():
    w = StubWrapper(checkpoint="stub", device="cpu")
    x = torch.randn(512)           # 1-D input → auto-batched to (1, 512)
    out = w.get_embeddings(x)
    assert out.embeddings.shape[0] == 1


def test_mean_pool():
    emb = torch.ones(3, 10, 16)
    pooled = TSFMWrapper._mean_pool(emb)
    assert pooled.shape == (3, 16)
    assert torch.allclose(pooled, torch.ones(3, 16))


def test_embedding_output_fields():
    emb = torch.zeros(2, 5, 32)
    out = EmbeddingOutput(embeddings=emb, pooled=emb.mean(1), layer_idx=3)
    assert out.layer_idx == 3
    assert out.meta == {}
