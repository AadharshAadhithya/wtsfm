#!/usr/bin/env python3
"""Parse TSFM benchmark runs into a wide-format CSV table.

CSV schema:
    Model | Dataset | Task | Metric | lp | lora | ft2

Each (model, dataset, task, metric) combination is one row,
with the metric value filled in for each adaptation mode column.

Task/metric mapping (derived from dataset + result.json metric keys):
    interf  -> Interference Detection       : det_acc
    interf  -> Interference Classification  : mod_acc
    radcom  -> Radar Signal Classification  : sig_acc
    radcom  -> Modulation Classification    : mod_acc
    rml     -> Modulation Classification    : pca
    rfp     -> RF Fingerprinting            : pca
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Task / metric / display-name definitions (matching the reference table)
# ---------------------------------------------------------------------------

# (dataset_slug, metric_key) -> (Task display name, Metric display name)
TASK_METRIC_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("interf", "det_acc"): ("Interference Detection",      "Interference Detection Accuracy"),
    ("interf", "mod_acc"): ("Interference Classification", "Interference Classification Accuracy"),
    ("radcom", "sig_acc"): ("Radar Signal Classification", "Radar Signal Classification Accuracy"),
    ("radcom", "mod_acc"): ("Modulation Classification",   "Modulation Classification Accuracy"),
    ("rml",    "pca"):     ("Modulation Classification",   "Mean Per-Class Accuracy"),
    ("rfp",    "pca"):     ("RF Fingerprinting",           "Mean Per-Class Accuracy"),
}

# Human-readable dataset names
DATASET_DISPLAY: dict[str, str] = {
    "interf": "ICARUS",
    "radcom": "RADCOM",
    "rml":    "RML2022",
    "rfp":    "POWDER",
}

# Human-readable model names
MODEL_DISPLAY: dict[str, str] = {
    "chronos2": "Chronos-2",
    "timesfm":  "TimesFM",
    "patchtst": "PatchTST",
    "moirai2":  "Moirai-2",
    "timer_s1": "Timer-S1",
    "sundial":  "Sundial",
    "tirex":    "TiRex",
    "toto":     "Toto",
}

MODES = ["lp", "lora", "ft2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _slug_dataset(slug: str) -> str:
    """Extract dataset token from run folder slug: {dataset}_{model}_{mode}_s{seed}."""
    return slug.split("_")[0]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse tsfm benchmark result.json files into a wide-format CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="tsfm_runs/benchmark_preprocessed",
        help="Directory containing run subfolders with result.json files.",
    )
    parser.add_argument(
        "--output-csv",
        default="tsfm_runs/benchmark_preprocessed_summary.csv",
        help="Path of generated CSV file.",
    )
    return parser.parse_args()


def load_results(input_dir: Path) -> dict:
    """
    Returns a nested dict:
        results[model][dataset_slug][mode][metric_key] = value
    """
    results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for result_path in sorted(input_dir.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text())
        except Exception as exc:
            print(f"[warn] could not parse {result_path}: {exc}")
            continue

        model   = payload.get("model", "unknown")
        mode    = payload.get("mode",  "unknown")
        dataset = _slug_dataset(result_path.parent.name)
        metrics = payload.get("metrics", {})

        for key, val in metrics.items():
            results[model][dataset][mode][key] = val

    return results


def build_rows(results: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for model_slug in sorted(results):
        model_display = MODEL_DISPLAY.get(model_slug, model_slug)

        for dataset_slug in sorted(results[model_slug]):
            dataset_display = DATASET_DISPLAY.get(dataset_slug, dataset_slug)
            modes_data = results[model_slug][dataset_slug]

            for (ds, metric_key), (task_name, metric_name) in TASK_METRIC_MAP.items():
                if ds != dataset_slug:
                    continue

                # Only emit a row if at least one mode has this metric
                mode_values = {
                    mode: modes_data.get(mode, {}).get(metric_key)
                    for mode in MODES
                }
                if all(v is None for v in mode_values.values()):
                    continue

                row: dict[str, str] = {
                    "Model":   model_display,
                    "Dataset": dataset_display,
                    "Task":    task_name,
                    "Metric":  metric_name,
                }
                for mode in MODES:
                    val = mode_values[mode]
                    row[mode] = _fmt(val) if val is not None else ""

                rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    input_dir  = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    results = load_results(input_dir)
    rows    = build_rows(results)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Model", "Dataset", "Task", "Metric", "lp", "lora", "ft2"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} rows to {output_csv}")
    _print_table(rows, fieldnames)


def _print_table(rows: list[dict], fieldnames: list[str]) -> None:
    col_widths = {f: len(f) for f in fieldnames}
    for row in rows:
        for f in fieldnames:
            col_widths[f] = max(col_widths[f], len(row.get(f, "")))

    sep = "+-" + "-+-".join("-" * col_widths[f] for f in fieldnames) + "-+"
    header = "| " + " | ".join(f.ljust(col_widths[f]) for f in fieldnames) + " |"

    print(sep)
    print(header)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(row.get(f, "").ljust(col_widths[f]) for f in fieldnames) + " |"
        print(line)
    print(sep)


if __name__ == "__main__":
    main()
