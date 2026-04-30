# TSFM Eval Harness

Unified embedding extraction harness for modern Time Series Foundation Models.

Given a time series tensor, get patch/token embeddings from any supported model
at any layer — ready for linear probing, LoRA fine-tuning, or full fine-tuning.

---

## Supported Models

| Key | Model | Source |
|-----|-------|--------|
| `patchtst` | PatchTST | HuggingFace `transformers` |
| `chronos2` | Chronos-2 (Amazon) | `chronos-forecasting` package |
| `timesfm` / `timesfm25` | TimesFM 2.0 / 2.5 (Google) | HuggingFace `transformers>=4.52` |
| `sundial` / `timer3` | Sundial / Timer 3.0 (THUML) | HuggingFace (trust_remote_code) |
| `timers1` / `timer_s1` | Timer-S1 (THUML) | HuggingFace (trust_remote_code) |
| `moirai` / `moirai2` | MOIRAI-2 (Salesforce) | `uni2ts` package |
| `tirex` | TiRex (NX-AI, xLSTM) | `tirex-ts` package |
| `toto` | TOTO-2 (Datadog) | GitHub source install |

---

## Installation

### 1. Core harness

```bash
git clone <this-repo>
cd tsfm-evalharness
pip install -e .
```

### 2. Per-model dependencies

Install only what you need:

```bash
# PatchTST (already covered by core transformers dep)
pip install "transformers>=4.40"

# Chronos-2
pip install "chronos-forecasting>=2.0"

# TimesFM 2.5
pip install "transformers>=4.52"

# Sundial / Timer-S1
pip install "transformers>=4.40"
# (uses trust_remote_code=True — no extra package needed)

# MOIRAI-2
pip install "uni2ts"

# TiRex
pip install "tirex-ts"

# TOTO-2
pip install "toto @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
```

Or install everything at once (may take a while):

```bash
pip install -e ".[all]"
# Then manually add TOTO and MOIRAI since they're not on PyPI cleanly:
pip install "uni2ts"
pip install "toto @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
```

---

## Model Cache

All models use the **HuggingFace Hub cache** at:

```
~/.cache/huggingface/hub/
```

You can override this with an environment variable:

```bash
export HF_HOME=/path/to/your/cache
# or more specifically:
export HF_HUB_CACHE=/scratch/models/hf_cache
```

Models are downloaded **lazily** — nothing is fetched until you first call
`.get_embeddings()` (or access `.model`). Subsequent calls reuse the cache.

To pre-download a model without running inference:

```python
from harness import ModelRegistry
w = ModelRegistry.load("chronos2", device="cpu")
_ = w.model   # triggers download now
```

**Approximate model sizes on disk:**

| Model | Checkpoint | Size |
|-------|-----------|------|
| PatchTST | namctin/patchtst_etth1_pretrain | ~20 MB |
| Chronos-2 small | amazon/chronos-bolt-small | ~80 MB |
| TimesFM 2.0 | google/timesfm-2.0-500m-pytorch | ~1 GB |
| Sundial base | thuml/sundial-base-128m | ~500 MB |
| Timer-S1 | thuml/timer-base-84m | ~330 MB |
| MOIRAI-2 small | Salesforce/moirai-2.0-R-small | ~100 MB |
| TiRex | NX-AI/TiRex | ~140 MB |
| TOTO-2 | Datadog/Toto-2.0-22m | ~90 MB |

---

## Usage

### Basic: get embeddings from one model

```python
import torch
from harness import ModelRegistry

# Load model (downloaded on first use, cached after)
wrapper = ModelRegistry.load("chronos2", device="cuda")

# Univariate: (batch, seq_len)
x = torch.randn(4, 512)
out = wrapper.get_embeddings(x)

print(out.embeddings.shape)  # (4, n_tokens, hidden_dim)
print(out.pooled.shape)      # (4, hidden_dim)  — mean-pooled, ready for a linear head
```

### Choose a specific layer

```python
# Last layer (default)
out = wrapper.get_embeddings(x, layer_idx=-1)

# First transformer layer
out = wrapper.get_embeddings(x, layer_idx=0)

# Middle layer
out = wrapper.get_embeddings(x, layer_idx=wrapper.num_layers // 2)
```

### Multivariate input

```python
# (batch, seq_len, n_vars)
x_mv = torch.randn(4, 512, 7)
out = wrapper.get_embeddings(x_mv)
# embeddings shape: (4, n_vars * n_tokens, hidden_dim)
```

### Use a custom checkpoint

```python
wrapper = ModelRegistry.load(
    "patchtst",
    checkpoint="my-org/my-finetuned-patchtst",
    device="cuda",
)
```

### Load multiple models at once

```python
wrappers = ModelRegistry.load_many(
    ["patchtst", "chronos2", "sundial"],
    device="cuda",
)

for name, w in wrappers.items():
    out = w.get_embeddings(x)
    print(f"{name}: {out.pooled.shape}")
```

### See all available model keys

```python
print(ModelRegistry.available())
# ['chronos2', 'moirai', 'moirai2', 'patchtst', 'sundial',
#  'timer3', 'timer_s1', 'timers1', 'timesfm', 'timesfm25', 'tirex', 'toto']
```

---

## Output format

`get_embeddings()` always returns an `EmbeddingOutput` dataclass:

```python
@dataclass
class EmbeddingOutput:
    embeddings: torch.Tensor   # (batch, n_tokens, hidden_dim) — raw token sequence
    pooled:     torch.Tensor   # (batch, hidden_dim)           — mean-pooled
    layer_idx:  int            # which layer was extracted
    meta:       dict           # model-specific info (n_patches, n_vars, etc.)
```

`embeddings` is always on **CPU** regardless of which device the model runs on.

---

## What's next

The harness gives you embeddings. The intended downstream workflows are:

- **Linear probing**: fit a `sklearn` linear model or a small MLP on `out.pooled`
- **LoRA fine-tuning**: attach `peft` LoRA adapters to the backbone, freeze the rest, train end-to-end
- **Full fine-tuning**: unfreeze the backbone and fine-tune everything
- **Zero-shot evaluation**: directly use `out.pooled` with a task-specific head, no training

These are not in the harness yet — the harness is responsible only for the
embedding extraction step.

---

## Troubleshooting

**Hook-based models (Sundial, MOIRAI, TiRex, TOTO) throw `AttributeError`**

The wrapper can't find the transformer/LSTM layers in the model's module tree.
Run:

```python
for name, _ in wrapper.model.named_modules():
    print(name)
```

Then update the `_get_*_layers()` / `_get_*_blocks()` method in the relevant
`harness/models/<model>.py` file with the correct attribute path.

**`ImportError` for a model**

Install the per-model dependency listed in the table above.

**Out of memory**

Use `device="cpu"` or a smaller checkpoint variant (e.g., `chronos-bolt-tiny`
instead of `chronos-bolt-base`).
