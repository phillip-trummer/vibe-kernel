#!/usr/bin/env python3
"""Turn a K-Search or AdaExplore usage log into the shared curve.csv.

    uv run python scripts/parse_usage_log.py <run>/usage.jsonl -o <run>/curve.csv

Both harnesses append one JSONL event per LLM call and per evaluation, stamped
with the run's cumulative token totals. They agree on the token buckets but not
on where the metrics sit: K-Search puts them on the event, AdaExplore nests them
under ``runtime_stats``. This reads either and writes the same columns
parse_claude_log.py produces, so plot_eval_runs.py can plot all three harnesses
side by side.

Geomeans are computed from the per-workload leaves when the harness did not
report them, and only when every workload has a usable value — a geomean over
the subset that happened to pass would flatter a kernel that failed the hard
shapes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

# USD per million tokens (Opus rates), matching parse_claude_log.py so a run is
# priced identically whichever harness produced it.
PRICE_INPUT = 5.0
PRICE_CACHED_INPUT = 0.50
PRICE_OUTPUT = 25.0

FIELDS = [
    "eval_index",
    "output_tokens",
    "logical_tokens",
    "compute_tokens",
    "cost_usd",
    "round",
    "status",
    "geomean_speedup_factor",
    "geomean_latency_ms",
    "representative_geomean_speedup_factor",
    "representative_geomean_latency_ms",
    "mean_speedup_factor",
    "score",
    "score_name",
    "num_workloads",
]


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _geomean(values: list[Any]) -> float | None:
    """Geomean over positive values, or None if any value is missing/unusable."""
    if not values:
        return None
    usable = [_number(v) for v in values]
    if any(v is None or v <= 0 for v in usable):
        return None
    return math.exp(sum(math.log(v) for v in usable) / len(usable))


def _metrics(event: dict[str, Any]) -> dict[str, Any]:
    """Metrics live on the event (K-Search) or under runtime_stats (AdaExplore)."""
    stats = event.get("runtime_stats")
    stats = stats if isinstance(stats, dict) else {}
    return {**stats, **{k: v for k, v in event.items() if v is not None}}


def _leaves(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    workloads = metrics.get("workloads")
    return [w for w in workloads if isinstance(w, dict)] if isinstance(workloads, list) else []


def _status(event: dict[str, Any]) -> str:
    """One status string from either harness's flags."""
    status = event.get("status")
    if isinstance(status, str) and status:
        return status
    if not event.get("compiled", True):
        return "COMPILE_ERROR"
    return "passed" if event.get("correctness") else "failed"


def _cost_usd(event: dict[str, Any], prices: tuple[float, float, float]) -> float | None:
    price_input, price_cached, price_output = prices
    inputs = _number(event.get("cumulative_input_tokens"))
    cached = _number(event.get("cumulative_cached_input_tokens")) or 0.0
    output = _number(event.get("cumulative_output_tokens"))
    if inputs is None or output is None:
        return None
    return round(
        ((inputs - cached) * price_input + cached * price_cached + output * price_output) / 1e6,
        4,
    )


def _row(
    index: int,
    event: dict[str, Any],
    prices: tuple[float, float, float],
    representatives: set[str],
) -> dict[str, Any]:
    metrics = _metrics(event)
    leaves = _leaves(metrics)
    selected = [leaf for leaf in leaves if str(leaf.get("uuid")) in representatives]

    speedups = [leaf.get("speedup") for leaf in leaves]
    latencies = [leaf.get("latency_ms") for leaf in leaves]
    return {
        "eval_index": index,
        "output_tokens": event.get("cumulative_output_tokens"),
        "logical_tokens": event.get("cumulative_total_tokens"),
        "compute_tokens": event.get("cumulative_compute_tokens"),
        "cost_usd": _cost_usd(event, prices),
        "round": event.get("round", event.get("step")),
        "status": _status(event),
        # Prefer what the harness reported; fall back to the leaves.
        "geomean_speedup_factor": (
            _number(metrics.get("geomean_speedup_factor"))
            or _number(metrics.get("geomean_speedup"))
            or _geomean(speedups)
        ),
        "geomean_latency_ms": (
            _number(metrics.get("geomean_latency_ms")) or _geomean(latencies)
        ),
        "representative_geomean_speedup_factor": _geomean(
            [leaf.get("speedup") for leaf in selected]
        ),
        "representative_geomean_latency_ms": _geomean(
            [leaf.get("latency_ms") for leaf in selected]
        ),
        "mean_speedup_factor": (
            _number(metrics.get("mean_speedup_factor"))
            or _number(metrics.get("speedup_factor"))
        ),
        "score": _number(metrics.get("score")),
        "score_name": metrics.get("score_name"),
        "num_workloads": metrics.get("num_workloads") or metrics.get("total_workloads"),
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error: invalid JSON at {path}:{line_number}: {exc}")
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log", type=Path, help="usage.jsonl from K-Search or AdaExplore.")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output CSV (default: curve.csv next to the log).")
    parser.add_argument("--representative", nargs="+", default=(),
                        help="Workload UUIDs forming the representative subset.")
    parser.add_argument("--price-input", type=float, default=PRICE_INPUT,
                        help="USD per 1M uncached input tokens.")
    parser.add_argument("--price-cached-input", type=float, default=PRICE_CACHED_INPUT,
                        help="USD per 1M cached input tokens.")
    parser.add_argument("--price-output", type=float, default=PRICE_OUTPUT,
                        help="USD per 1M output tokens.")
    args = parser.parse_args(argv)

    log = args.log.expanduser().resolve()
    if not log.is_file():
        raise SystemExit(f"Error: usage log not found: {log}")
    events = read_events(log)
    evals = [event for event in events if event.get("event") == "eval"]
    if not evals:
        raise SystemExit(f"Error: no eval events in {log}")

    prices = (args.price_input, args.price_cached_input, args.price_output)
    representatives = set(args.representative)
    output = args.output or log.parent / "curve.csv"
    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        for index, event in enumerate(evals, 1):
            writer.writerow(_row(index, event, prices, representatives))

    calls = sum(1 for event in events if event.get("event") == "llm_call")
    print(f"Read {len(evals)} evaluations and {calls} LLM calls from {log}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
