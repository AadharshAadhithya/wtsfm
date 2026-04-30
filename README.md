# Wireless TSFM Benchmark

Benchmarks general-purpose **time-series foundation models (TSFMs)** against **WavesFM** — a wireless-domain-specific foundation model — on IQ signal classification tasks. All three systems are evaluated under the same three adaptation modes (linear probe, partial fine-tuning, LoRA) using the exact same training loop, so results are directly comparable.

---

## Overview

| System | What it is | Adaptation |
|--------|-----------|------------|
| **WavesFM** | Modality-adaptive ViT pre-trained on wireless signals | lp / ft2 / lora |
| **TSFMs** | 8 general time-series FMs (Chronos-2, TimesFM, PatchTST, …) | lp / ft2 / lora |

**Tasks (IQ-domain only)**

| Key | Dataset file | Signal type |
|-----|-------------|-------------|
| `radcom` | `radcom.h5` | Radio communications modulation |
| `rml` | `rml22.h5` | RadioML 2022 modulation recognition |
| `rfp` | `rfp.h5` | RF fingerprinting |
| `interf` | `icarus.h5` | Interference classification |

---

## Repository structure

```
wireless-tsfms/
├── wavesfm/                   # WavesFM codebase (unchanged)
│   ├── main_finetune.py       # WavesFM fine-tuning entry point
│   ├── engine.py              # train_one_epoch / evaluate (reused by TSFMs)
│   └── ...
├── tsfm-evalharness/          # TSFM embedding harness
│   ├── harness/
│   │   ├── base.py            # TSFMWrapper ABC + EmbeddingOutput dataclass
│   │   ├── registry.py        # ModelRegistry.load("chronos2", device="cuda")
│   │   └── models/
│   │       ├── chronos.py     # Chronos-2 (multivariate via group_ids)
│   │       ├── timesfm.py     # TimesFM 2.5
│   │       ├── patchtst.py    # PatchTST
│   │       ├── sundial.py     # Sundial / Timer 3.0 (hook-based)
│   │       ├── timer_s1.py    # Timer-S1 84M (same as Sundial)
│   │       ├── moirai.py      # MOIRAI-2 (hook-based)
│   │       ├── tirex.py       # TiRex xLSTM (hook-based)
│   │       └── toto.py        # TOTO-2 (hook-based)
│   └── pyproject.toml
├── data/preprocessed/         # Put .h5 files here
├── run_tsfm_probe.py          # TSFM downstream training script
├── run_tsfm_benchmark.sh      # Sweep: all TSFMs × modes × seeds × tasks
└── run_finetuning_wavesfm.sh  # Sweep: WavesFM × modes × seeds × tasks
```

---

## Installation

### 1. Python environment

```bash
conda create -n wireless-tsfms python=3.11
conda activate wireless-tsfms
```

### 2. PyTorch

Install the wheel matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. WavesFM dependencies

```bash
pip install timm h5py numpy pandas scipy tqdm
```

### 4. TSFM eval harness

```bash
pip install -e tsfm-evalharness/
```

This installs the core harness with `transformers`, `huggingface_hub`, and `numpy`. Each model has optional extra dependencies — install only what you need:

| Model key | Extra install |
|-----------|--------------|
| `patchtst` | *(included in core `transformers`)* |
| `timesfm` | `pip install -U transformers` (≥4.52) |
| `sundial` / `timer_s1` | *(included in core `transformers`, uses `trust_remote_code`)* |
| `chronos2` | `pip install chronos-forecasting>=2.0` |
| `moirai2` | `pip install uni2ts` |
| `tirex` | `pip install tirex-ts` |
| `toto` | `pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"` |

To install everything at once:

```bash
pip install -e "tsfm-evalharness/[all]"
pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
```

### 5. LoRA support

```bash
pip install peft
```

Required only if running `--mode lora`.

---

## Data

Place the preprocessed `.h5` files under `data/preprocessed/`:

```
data/preprocessed/
├── radcom.h5
├── rml22.h5
├── rfp.h5
└── icarus.h5
```

Each file is consumed by `wavesfm/data.py` → `build_datasets()`, which returns PyTorch `Dataset` objects that yield `(signal, label)` pairs where `signal` is shape `(2, T)` — I and Q channels.

---

## Running the benchmark

### TSFM sweep (all models × modes × seeds × tasks)

```bash
bash run_tsfm_benchmark.sh
```

Key environment overrides:

```bash
# Run only Chronos-2 and TimesFM in linear-probe mode, one seed
MODELS="chronos2 timesfm" MODES="lp" SEEDS="0" bash run_tsfm_benchmark.sh

# Custom data root and output directory
DATA_ROOT=/mnt/data/preprocessed OUTPUT_ROOT=/results/tsfm bash run_tsfm_benchmark.sh
```

Results land in `tsfm_runs/{task}_{model}_{mode}_s{seed}/result.json`.

### WavesFM sweep (same modes and tasks)

```bash
CKPT_PATH=/path/to/wavesfm_pretrained.pth bash run_finetuning_wavesfm.sh
```

Results land in `wavesfm_runs/wavesfm_{mode}/{task}/s{seed}/`.

### Single run (for debugging / custom checkpoints)

```bash
python run_tsfm_probe.py \
    --task rml \
    --train-data data/preprocessed/rml22.h5 \
    --model chronos2 \
    --mode lp \
    --seed 0
```

```bash
python run_tsfm_probe.py \
    --task interf \
    --train-data data/preprocessed/icarus.h5 \
    --model timesfm \
    --mode lora \
    --lora-rank 32 --lora-alpha 64 \
    --checkpoint google/timesfm-2.0-500m-pytorch
```

---

## Adaptation modes

| Mode | What trains | WavesFM equivalent |
|------|------------|-------------------|
| `lp` | Linear head only; entire encoder frozen | `--freeze-encoder` (default) |
| `ft2` | Second half of encoder blocks + head; first half frozen | `--frozen-blocks 6` |
| `lora` | LoRA A/B matrices (R=32, α=64) on Q/V projections + head | `--lora --lora-rank 32 --lora-alpha 64` |

All modes use the same training loop, optimizer (AdamW + cosine schedule + GradScaler), and task-specific hyperparameters as WavesFM.

**Task-specific defaults** (same as `wavesfm/run_finetune_all.py`):

| Task | Epochs | Batch size | Label smoothing | Grad accum | Stratified split |
|------|--------|------------|-----------------|------------|-----------------|
| `rfp` | 10 | 256 | 0.1 | 1 | No |
| `interf` | 35 | 256 | 0.02 | 2 | Yes |
| `rml` | 50 | 2048 | — | 1 | No |
| `radcom` | 50 | 2048 | — | 1 | No |

---

## Input / output contract

### What goes into the TSFMs

IQ datasets return `(signal, label)` where `signal` is `(2, T)` — two channels (I and Q), `T` time steps.

`TSFMProbeModel.forward()` permutes this to `(B, T, 2)` before passing to the harness:

```
Raw dataset batch:  (B, 2, T)   — channels first, as stored in .h5
                        ↓  permute(0,2,1)
TSFM input:         (B, T, 2)   — (batch, seq_len, n_vars=2)
```

The TSFMs see two "variates" representing I and Q. No normalisation is applied at the harness level — values are passed as raw float32.

### What comes out

Every wrapper returns an `EmbeddingOutput`:

```python
@dataclass
class EmbeddingOutput:
    embeddings: torch.Tensor   # (B, n_tokens, D)  — full token sequence
    pooled:     torch.Tensor   # (B, D)             — mean-pooled over tokens
    layer_idx:  int            # which layer was extracted
    meta:       dict           # model-specific info (n_vars, n_patches, …)
```

`pooled` is passed directly to the linear classification head: `head(out.pooled) → (B, num_classes)`.

Token counts and embedding dimensions per model (approximate, depends on `T` and checkpoint):

| Model | Architecture | `n_tokens` | `D` |
|-------|-------------|-----------|-----|
| Chronos-2 | Encoder, group attention | n_patches | 512 |
| TimesFM 2.5 | Decoder-only | n_patches | 512+ |
| PatchTST | Encoder Transformer | n_vars × n_patches | 512 |
| Sundial / Timer 3.0 | Decoder-only (hook) | n_patches | 1024 |
| Timer-S1 84M | Decoder-only (hook) | n_patches | 1024 |
| MOIRAI-2 | Decoder-only (hook) | n_patches | 384+ |
| TiRex | xLSTM (hook) | T / patch_size | varies |
| TOTO-2 | Temporal+variate attn (hook) | n_patches | 512 |

The actual `D` for the loaded run is printed at startup:

```
[model] embed_dim=512
```

### Multivariate handling per model

| Model | Strategy |
|-------|---------|
| **Chronos-2** | Native: I and Q passed as `group_ids`-linked variates; encoder mixes them via group attention before pooling |
| **PatchTST** | Native: `(B, T, 2)` → `(B, 2, n_patches, D)`; I and Q patches concatenated into token sequence then mean-pooled |
| **TimesFM** | Channel-independent: flattened to `(2B, T)`, embeddings reshaped back and mean-pooled |
| **TOTO-2** | Native variate-attention: handles `(B, T, 2)` directly |
| **Sundial / Timer / MOIRAI / TiRex** | Univariate-only internally; mean-pool across I/Q run separately if needed |

---

## Changes made to the harness

The original harness was designed for inference-only embedding extraction. Several changes were necessary to support fine-tuning (ft2 / lora modes) and to keep the comparison consistent with WavesFM's PyTorch training loop.

### 1. Removed `torch.no_grad()` from all wrappers

**Files:** `chronos.py`, `timesfm.py`, `patchtst.py`, `sundial.py`, `moirai.py`, `tirex.py`, `toto.py`

The original wrappers wrapped their forward pass in `with torch.no_grad():`. This silently disconnects the computation graph, making gradient-based fine-tuning impossible. Removed from all wrappers so that ft2 and lora gradients flow back through the encoder.

For lp (linear probe) the encoder is frozen via `requires_grad=False`, so no gradients reach it regardless.

### 2. Removed `.detach()` from hook callbacks

**Files:** `sundial.py`, `moirai.py`, `tirex.py`, `toto.py`

Hook-based wrappers (models that don't expose `output_hidden_states`) captured the hidden state with:

```python
captured.append(h.detach())   # ← breaks autograd graph
```

Changed to:

```python
captured.append(h)            # ← keeps tensor connected to graph
```

### 3. Removed `.cpu()` from `EmbeddingOutput` fields

**Files:** `chronos.py`, `timesfm.py`, `patchtst.py`, `sundial.py`, `moirai.py`, `tirex.py`, `toto.py`

Original wrappers returned embeddings on CPU (`.cpu()`). A GPU→CPU copy mid-graph is extremely slow under autocast and breaks mixed-precision training. Removed: tensors now stay on the model's device. The training loop in `run_tsfm_probe.py` places the head on the same device, so no cross-device mismatch occurs.

### 4. Chronos-2 rewritten for multivariate + group attention

**File:** `chronos.py`

The original wrapper called `Chronos2Pipeline.embed()`, which only accepts a list of univariate 1-D tensors. For IQ data (two variates), we need Chronos-2's group-attention mechanism so I and Q can cross-attend before pooling.

The new implementation calls the model internals directly:

```python
patched, attn_mask, _ = inner._prepare_patched_context(x_flat, context_mask=None)
input_embeds = inner.input_patch_embedding(patched)
enc_out = inner.encoder(
    attention_mask=attn_mask,
    inputs_embeds=input_embeds,
    group_ids=group_ids,   # ← variates of the same sample share one id
    output_attentions=False,
)
```

`group_ids` is constructed so I and Q from the same sample share the same group id:
```
x (B,T,2) → flatten → x_flat (2B, T)
group_ids = [0,0, 1,1, ..., B-1, B-1]
```

### 5. New `run_tsfm_probe.py` — PyTorch training loop

The original benchmark used scikit-learn `LogisticRegression` on frozen embeddings. This is not comparable to WavesFM's AdamW + cosine schedule + GradScaler training.

`run_tsfm_probe.py` introduces `TSFMProbeModel`, a `nn.Module` that:

- Accepts `(B, 2, T)` — the raw IQ batch from the datasets (same interface as WavesFM)
- Permutes to `(B, T, 2)` and calls the harness wrapper
- Passes `out.pooled` to a `nn.Linear` head
- Returns `(B, num_classes)` logits

This makes it a drop-in replacement for WavesFM's ViT, so `engine.train_one_epoch()` and `engine.evaluate()` from `wavesfm/engine.py` work unchanged.

---

## Output files

After each run, `run_tsfm_probe.py` writes to `{output_dir}/{task}_{model}_{mode}_s{seed}/`:

| File | Contents |
|------|---------|
| `result.json` | Final metrics, config, timing |
| `best.pth` | Checkpoint at best validation metric |
| `checkpoint_NNN.pth` | Periodic / final checkpoint |
| `train.log` | Redirected stdout from the shell sweep |
| `log.jsonl` | Per-epoch metrics (train loss, val accuracy, LR, …) |
