"""
Central model registry.
Maps model name → (WrapperClass, default_checkpoint).
"""
from __future__ import annotations
from typing import Dict, Tuple, Type, Optional, Any

from .base import TSFMWrapper

_REGISTRY: Dict[str, Tuple[str, str]] = {
    # key: (wrapper_module_path, default_checkpoint)
    "patchtst":  ("harness.models.patchtst",  "namctin/patchtst_etth1_pretrain"),
    "chronos2":  ("harness.models.chronos",   "amazon/chronos-2"),
    "timesfm":   ("harness.models.timesfm",   "google/timesfm-2.0-500m-pytorch"),
    "timesfm25": ("harness.models.timesfm",   "google/timesfm-2.0-500m-pytorch"),
    "sundial":   ("harness.models.sundial",   "thuml/sundial-base-128m"),
    "timer3":    ("harness.models.sundial",   "thuml/sundial-base-128m"),
    "timer_s1":  ("harness.models.timer_s1",  "thuml/timer-base-84m"),
    "timers1":   ("harness.models.timer_s1",  "thuml/timer-base-84m"),
    "moirai2":   ("harness.models.moirai",    "Salesforce/moirai-2.0-R-small"),
    "moirai":    ("harness.models.moirai",    "Salesforce/moirai-2.0-R-small"),
    "tirex":     ("harness.models.tirex",     "NX-AI/TiRex"),
    "toto":      ("harness.models.toto",      "Datadog/Toto-2.0-22m"),
}

_CLASS_MAP: Dict[str, str] = {
    "harness.models.patchtst":  "PatchTSTWrapper",
    "harness.models.chronos":   "ChronosWrapper",
    "harness.models.timesfm":   "TimesFMWrapper",
    "harness.models.sundial":   "SundialWrapper",
    "harness.models.timer_s1":  "TimerS1Wrapper",
    "harness.models.moirai":    "MoiraiWrapper",
    "harness.models.tirex":     "TiRexWrapper",
    "harness.models.toto":      "TotoWrapper",
}


class ModelRegistry:
    """
    Factory for TSFM wrappers.

    Usage
    -----
    from harness import ModelRegistry

    # Load with default checkpoint
    wrapper = ModelRegistry.load("chronos2", device="cuda")

    # Load with custom checkpoint
    wrapper = ModelRegistry.load("patchtst", checkpoint="my-org/my-patchtst", device="cpu")

    # Get embeddings
    import torch
    x = torch.randn(4, 512)          # (batch=4, seq_len=512)
    out = wrapper.get_embeddings(x, layer_idx=-1)
    print(out.embeddings.shape)       # (4, n_tokens, hidden_dim)
    print(out.pooled.shape)           # (4, hidden_dim)
    """

    @staticmethod
    def available() -> list[str]:
        return sorted(_REGISTRY.keys())

    @staticmethod
    def load(
        name: str,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> TSFMWrapper:
        """
        Instantiate and return a wrapper for the named model.

        Parameters
        ----------
        name       : Model key, e.g. "chronos2", "patchtst", "sundial".
        checkpoint : Override the default HuggingFace / local checkpoint path.
        device     : Torch device string, e.g. "cpu", "cuda", "cuda:1".
        **kwargs   : Extra kwargs forwarded to the wrapper constructor.
        """
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(
                f"Unknown model {name!r}. Available: {available}"
            )

        module_path, default_ckpt = _REGISTRY[key]
        cls_name = _CLASS_MAP[module_path]

        import importlib
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("harness.models"):
                raise ModuleNotFoundError(
                    "Could not import TSFM wrapper modules (missing 'harness.models'). "
                    "Your local 'tsfm-evalharness' checkout appears incomplete. "
                    "Ensure 'tsfm-evalharness/harness/models/' exists and reinstall with "
                    "'pip install -e tsfm-evalharness/'."
                ) from exc
            raise
        cls: Type[TSFMWrapper] = getattr(module, cls_name)

        ckpt = checkpoint or default_ckpt
        return cls(checkpoint=ckpt, device=device, **kwargs)

    @staticmethod
    def load_many(
        names: list[str],
        device: str = "cpu",
        checkpoints: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, TSFMWrapper]:
        """
        Load multiple models at once.

        Parameters
        ----------
        names      : List of model keys.
        device     : Shared device for all models.
        checkpoints: Optional dict mapping name → checkpoint override.
        """
        checkpoints = checkpoints or {}
        return {
            name: ModelRegistry.load(
                name,
                checkpoint=checkpoints.get(name),
                device=device,
                **kwargs,
            )
            for name in names
        }
