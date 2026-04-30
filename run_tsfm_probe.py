#!/usr/bin/env python3
"""
TSFM downstream benchmark — IQ tasks, three adaptation modes.

Uses the same training loop, optimizer, schedule, and evaluation metrics as
wavesfm/main_finetune.py so the comparison is apples-to-apples.

Modes
-----
lp     — linear probe: encoder frozen, only the linear head is trained.
ft2    — partial fine-tune: freeze first half of transformer/LSTM blocks,
         train the rest + head.
lora   — LoRA (R=32 α=64) injected into all attention Q/V projections via peft;
         encoder otherwise frozen.

In all three modes the training loop, loss, optimizer (AdamW + cosine schedule
+ GradScaler), batch sizes, and epochs match the WavesFM defaults for each
task so results are directly comparable.

Input contract
--------------
Datasets return (B, 2, T).  TSFMProbeModel permutes to (B, T, 2) before
passing to the TSFM wrapper, matching the harness (B, seq_len, n_vars) API.

Usage
-----
    python run_tsfm_probe.py \\
        --task rml --train-data /data/rml22.h5 \\
        --model chronos2 --mode lp

    python run_tsfm_probe.py \\
        --task radcom --train-data /data/radcom.h5 \\
        --model timesfm --mode ft2 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "wavesfm"))
sys.path.insert(0, str(ROOT / "tsfm-evalharness"))

from data import build_datasets                                    # wavesfm
from engine import evaluate                                        # wavesfm
from utils import cosine_schedule, apply_lr, set_seed, JsonlLogger, pretty_dict  # wavesfm
from harness.registry import ModelRegistry                         # tsfm-evalharness

IQ_TASKS = ("radcom", "rml", "rfp", "interf")

# Task-specific defaults matching wavesfm/run_finetune_all.py
TASK_EPOCHS  = {"rfp": 10, "interf": 35, "rml": 50, "radcom": 50}
TASK_BATCH   = {"rml": 2048, "radcom": 2048}
DEFAULT_BATCH = 256

STRATIFIED_TASKS = {"interf"}
SMOOTH_TASKS     = {"rfp": 0.1, "interf": 0.02}
ACCUM_TASKS      = {"interf": 2}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TSFM downstream benchmark — IQ tasks.")
    # data (identical flags to main_finetune.py)
    p.add_argument("--task",             required=True, choices=IQ_TASKS)
    p.add_argument("--train-data",       dest="train_path", required=True)
    p.add_argument("--val-data",         dest="val_path",   default=None)
    p.add_argument("--val-split",        type=float, default=0.2)
    p.add_argument("--stratified-split", action="store_true")
    p.add_argument("--seed",             type=int, default=42)
    # model
    p.add_argument("--model",      required=True,
                   help="TSFM name — chronos2 / timesfm / patchtst / sundial / "
                        "timer_s1 / moirai2 / tirex / toto")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--layer-idx",  type=int, default=-1)
    # mode
    p.add_argument("--mode",       default="lp", choices=["lp", "ft2", "lora"],
                   help="lp=linear probe, ft2=partial fine-tune, lora=LoRA adapters")
    p.add_argument("--lora-rank",  type=int, default=32)
    p.add_argument("--lora-alpha", type=float, default=64.0)
    # training (defaults match wavesfm/run_finetune_all.py)
    p.add_argument("--epochs",          type=int,   default=None,
                   help="Override default epochs for the task.")
    p.add_argument("--batch-size",      type=int,   default=None)
    p.add_argument("--accum-steps",     type=int,   default=None)
    p.add_argument("--blr",            type=float, default=1e-3,
                   help="Base LR; effective LR = blr * batch / 256.")
    p.add_argument("--min-lr",         type=float, default=1e-6)
    p.add_argument("--weight-decay",   type=float, default=0.05)
    p.add_argument("--warmup-epochs",  type=float, default=5.0)
    p.add_argument("--max-grad-norm",  type=float, default=None)
    p.add_argument("--smoothing",      type=float, default=None)
    # IO
    p.add_argument("--output-dir",  default="tsfm_runs")
    p.add_argument("--save-every",  type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# TSFMProbeModel — drop-in replacement for WavesFM's ViT
# ─────────────────────────────────────────────────────────────────────────────

class TSFMProbeModel(nn.Module):
    """
    TSFM encoder + linear classification head.

    forward() accepts (B, 2, T) — the raw IQ batch format from the datasets —
    permutes to (B, T, 2), passes through the TSFM wrapper's get_embeddings(),
    and returns (B, num_classes) logits.  This is the same interface that
    WavesFM models expose, so WavesFM's engine.py functions work unchanged.
    """

    def __init__(self, wrapper, embed_dim: int, num_classes: int):
        super().__init__()
        self.wrapper   = wrapper
        self.head      = nn.Linear(embed_dim, num_classes)
        self._embed_dim = embed_dim
        self._layer_idx = -1   # set by caller if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, T) — IQ channels first, as returned by the datasets
        x_tvn = x.float().permute(0, 2, 1)   # → (B, T, 2)
        out   = self.wrapper.get_embeddings(x_tvn, layer_idx=self._layer_idx)
        # out.pooled is on the wrapper's device; head must be on the same device
        return self.head(out.pooled)

    # ── freeze / unfreeze helpers ──────────────────────────────────────────

    def freeze_encoder(self) -> None:
        for p in self.wrapper.model.parameters():
            p.requires_grad = False

    def unfreeze_head(self) -> None:
        for p in self.head.parameters():
            p.requires_grad = True

    def freeze_encoder_partial(self) -> int:
        """
        Freeze the first half of detected transformer/LSTM blocks;
        unfreeze the second half.  Returns number of frozen blocks.
        """
        blocks = _get_encoder_blocks(self.wrapper)
        if not blocks:
            # Fallback: freeze first 50% of parameters by count
            params = list(self.wrapper.model.parameters())
            n_freeze = len(params) // 2
            for p in params[:n_freeze]:
                p.requires_grad = False
            for p in params[n_freeze:]:
                p.requires_grad = True
            return n_freeze

        n_freeze = max(1, len(blocks) // 2)
        for block in blocks[:n_freeze]:
            for p in block.parameters():
                p.requires_grad = False
        for block in blocks[n_freeze:]:
            for p in block.parameters():
                p.requires_grad = True
        return n_freeze

    def apply_lora(self, rank: int = 32, alpha: float = 64.0) -> None:
        """
        Inject LoRA adapters via peft.  Freezes the entire encoder first,
        then peft unfreezes only the LoRA A/B matrices.
        """
        try:
            from peft import get_peft_model, LoraConfig
        except ImportError:
            raise ImportError("pip install peft")

        self.freeze_encoder()

        # Detect common attention Q/V projection names; fall back to all-linear
        target = _detect_qv_modules(self.wrapper.model) or "all-linear"
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target,
            lora_dropout=0.0,
            bias="none",
        )
        lora_model = get_peft_model(self.wrapper.model, config)
        self.wrapper._model = lora_model   # replace model inside wrapper


def _get_encoder_blocks(wrapper) -> List[nn.Module]:
    """
    Try known attribute paths to find the list of transformer/LSTM blocks.
    Checks both the pipeline wrapper (e.g. Chronos2Pipeline) and the
    inner model (e.g. Chronos2Model).
    """
    candidates = [wrapper.model]
    # Unwrap one level (e.g. Chronos2Pipeline → Chronos2Model)
    if hasattr(wrapper.model, "model"):
        candidates.append(wrapper.model.model)

    search_paths = [
        "encoder.layers", "layers", "blocks",
        "transformer.h", "model.layers", "decoder.layers",
        "xlstm_stack.blocks", "backbone.blocks", "model.blocks",
        "model.xlstm_stack.blocks", "transformer.blocks",
    ]
    for root in candidates:
        for path in search_paths:
            obj = root
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)

    # Try wrapper-level helper methods
    for method in ["_get_decoder_layers", "_get_attention_blocks",
                   "_get_xlstm_blocks", "_get_transformer_layers"]:
        fn = getattr(wrapper, method, None)
        if fn:
            try:
                blocks = fn()
                if blocks:
                    return blocks
            except Exception:
                pass

    return []


def _detect_qv_modules(model: nn.Module) -> list[str] | None:
    """Return unique leaf module names that look like Q/V attention projections."""
    candidates = {"q_proj", "v_proj", "k_proj", "query", "value", "key",
                  "to_q", "to_v", "to_k", "q", "v", "k", "Wqkv", "c_attn"}
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    return list(found) if found else None


# ─────────────────────────────────────────────────────────────────────────────
# Embed-dim probe (one forward pass to learn the pooled size)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def probe_embed_dim(wrapper, sample: torch.Tensor) -> int:
    """Run one sample through the wrapper and return the pooled feature size."""
    x_tvn = sample.float().unsqueeze(0).permute(0, 2, 1)  # (1, T, 2)
    out = wrapper.get_embeddings(x_tvn)
    return int(out.pooled.shape[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (reuses wavesfm/engine.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model: TSFMProbeModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task_info,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    logger: JsonlLogger,
) -> dict:
    from engine import train_one_epoch

    # Two param groups: encoder (smaller LR) + head (full LR).
    # For lp the encoder is fully frozen so encoder_params is empty.
    encoder_params = [p for p in model.wrapper.model.parameters() if p.requires_grad]
    head_params    = list(model.head.parameters())

    eff_batch = args.batch_size * args.accum_steps
    lr = args.blr * eff_batch / 256

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr * 0.1})
    param_groups.append({"params": head_params, "lr": lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler(device="cuda")

    steps_per_epoch = max(1, len(train_loader))
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(args.warmup_epochs * steps_per_epoch)
    lr_schedule     = cosine_schedule(lr, args.min_lr, total_steps, warmup_steps)

    smooth = args.smoothing or 0.0
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=smooth if smooth > 0 else 0.0
    )

    # Best-metric tracking (same as main_finetune.py)
    best_metric = float("-inf")
    best_key    = "pca"

    save_every = args.save_every or args.epochs  # default: only save best

    for epoch in range(args.epochs):
        step_offset = epoch * steps_per_epoch
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, device, scaler, epoch,
            accum_steps=args.accum_steps,
            max_norm=args.max_grad_norm,
            lr_schedule=lr_schedule,
            start_step=step_offset,
            task_type=task_info.target_type,
            print_freq=20,
        )

        val_stats = evaluate(
            model, val_loader, device, criterion,
            args.task, task_info.target_type, task_info.num_outputs,
        )

        cur = val_stats.get(best_key)
        if cur is not None and cur > best_metric:
            best_metric = float(cur)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "best_metric": best_metric, "args": vars(args)},
                out_dir / "best.pth",
            )

        if (epoch + 1) % save_every == 0 or epoch + 1 == args.epochs:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch,
                 "best_metric": best_metric, "args": vars(args)},
                out_dir / f"checkpoint_{epoch:03d}.pth",
            )

        logger.write({
            "epoch": epoch, "lr": optimizer.param_groups[-1]["lr"],
            "train": train_stats, "val": val_stats,
            "best_metric": best_metric, "best_key": best_key,
        })

    print(f"[done] best {best_key}={best_metric:.4f}")
    return val_stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Apply task-level defaults (same as run_finetune_all.py)
    if args.epochs    is None: args.epochs    = TASK_EPOCHS.get(args.task, 100)
    if args.batch_size is None: args.batch_size = TASK_BATCH.get(args.task, DEFAULT_BATCH)
    if args.accum_steps is None: args.accum_steps = ACCUM_TASKS.get(args.task, 1)
    if not args.stratified_split and args.task in STRATIFIED_TASKS:
        args.stratified_split = True
    if args.smoothing is None:
        args.smoothing = SMOOTH_TASKS.get(args.task, 0.0)

    set_seed(args.seed)
    cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    slug    = f"{args.task}_{args.model}_{args.mode}_s{args.seed}"
    out_dir = Path(args.output_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = JsonlLogger(out_dir)

    # ── data ─────────────────────────────────────────────────────────────
    print(f"[data] task={args.task}  train={args.train_path}")
    train_ds, val_ds, task_info = build_datasets(
        args.task, args.train_path,
        val_path=args.val_path,
        val_split=args.val_split,
        stratified_split=args.stratified_split,
        seed=args.seed,
    )
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"num_classes={task_info.num_outputs}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── TSFM wrapper ──────────────────────────────────────────────────────
    print(f"[model] loading {args.model!r} on {device} ...")
    wrapper = ModelRegistry.load(
        args.model, checkpoint=args.checkpoint, device=str(device)
    )

    # Probe embed dim with one sample (no_grad — just a shape check)
    sample0, _ = train_ds[0]
    with torch.no_grad():
        embed_dim = probe_embed_dim(wrapper, sample0)
    print(f"[model] embed_dim={embed_dim}")

    # ── build TSFMProbeModel ──────────────────────────────────────────────
    probe = TSFMProbeModel(wrapper, embed_dim, task_info.num_outputs)
    probe._layer_idx = args.layer_idx

    if args.mode == "lp":
        probe.freeze_encoder()
        print("[mode] lp — encoder frozen, training head only")

    elif args.mode == "ft2":
        probe.freeze_encoder()                 # freeze all first
        n_frozen = probe.freeze_encoder_partial()  # then unfreeze second half
        probe.unfreeze_head()
        blocks = _get_encoder_blocks(wrapper)
        print(f"[mode] ft2 — frozen {n_frozen}/{len(blocks)} blocks, "
              f"training rest + head")

    elif args.mode == "lora":
        probe.apply_lora(rank=args.lora_rank, alpha=args.lora_alpha)
        probe.unfreeze_head()
        trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in probe.parameters())
        print(f"[mode] lora R={args.lora_rank} α={args.lora_alpha} — "
              f"trainable={trainable/1e6:.2f}M / {total/1e6:.2f}M")

    probe = probe.to(device)
    trainable = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in probe.parameters())
    print(f"[model] trainable={trainable/1e6:.2f}M  total={total/1e6:.2f}M")

    # ── train & evaluate ─────────────────────────────────────────────────
    print(f"[train] epochs={args.epochs}  batch={args.batch_size}  "
          f"accum={args.accum_steps}  blr={args.blr:.2e}")
    t0 = time.time()
    val_stats = train_and_evaluate(
        probe, train_loader, val_loader, task_info, args, device, out_dir, logger
    )
    elapsed = time.time() - t0

    # ── save result JSON ──────────────────────────────────────────────────
    result = {
        "task":       args.task,
        "model":      args.model,
        "mode":       args.mode,
        "checkpoint": args.checkpoint or "default",
        "seed":       args.seed,
        "epochs":     args.epochs,
        "n_train":    len(train_ds),
        "n_val":      len(val_ds),
        "embed_dim":  embed_dim,
        "elapsed_s":  round(elapsed, 1),
        "metrics":    val_stats,
    }
    out_path = out_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[done] → {out_path}")


if __name__ == "__main__":
    main()
