"""Benchmark one source parameter across several literal values."""

import json
from pathlib import Path

from kernel_tools.registry import registry

from ._workspace import SRC_DIR
from .benchmark_kernel import benchmark_kernel


SCHEMA = {
    "name": "benchmark_sweep",
    "description": (
        "Benchmark several values for one source parameter. old_string must "
        "match exactly once in filename. For each value, the tool replaces it "
        "with replacement_template after substituting {value}, runs the normal "
        "benchmark, and reports all results plus the best observed passing "
        "value by geomean speedup or latency. Every candidate starts from the "
        "same original source, and the original source is always restored. "
        "Use this for mechanical parameter sweeps; independently confirm the "
        "winner before retaining it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Source filename containing the parameter.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact source text to replace; it must be unique.",
            },
            "replacement_template": {
                "type": "string",
                "description": (
                    "Replacement source text containing the literal placeholder "
                    "{value}."
                ),
            },
            "values": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Values substituted into replacement_template.",
            },
            "scope": {
                "type": "string",
                "enum": ["smoke", "full"],
                "default": "smoke",
            },
        },
        "required": [
            "filename",
            "old_string",
            "replacement_template",
            "values",
        ],
    },
}


@registry.register(SCHEMA)
def benchmark_sweep(
    workspace: Path,
    filename: str,
    old_string: str,
    replacement_template: str,
    values: list[str],
    scope: str = "smoke",
) -> str:
    if scope not in ("smoke", "full"):
        return "Error: scope must be 'full' or 'smoke'."
    if not values:
        return "Error: values must not be empty."
    if not old_string:
        return "Error: old_string must not be empty."
    if "{value}" not in replacement_template:
        return "Error: replacement_template must contain {value}."

    src_dir = (workspace / SRC_DIR).resolve()
    file_path = (src_dir / filename).resolve()
    if src_dir not in file_path.parents:
        return f"Error: {filename!r} is not a valid source filename."
    if not file_path.is_file():
        return f"Error: {filename!r} not found."

    original = file_path.read_text()
    count = original.count(old_string)
    if count == 0:
        return "Error: old_string not found. Re-read the file and copy it exactly."
    if count > 1:
        return (
            f"Error: old_string matches {count} locations. Add surrounding "
            "context to make it unique."
        )

    results: list[dict] = []
    try:
        for value in values:
            replacement = replacement_template.replace("{value}", value)
            file_path.write_text(original.replace(old_string, replacement, 1))
            raw_result = benchmark_kernel(workspace, scope=scope)
            try:
                result = json.loads(raw_result)
            except json.JSONDecodeError:
                results.append({"value": value, "error": raw_result})
                continue
            if not isinstance(result, dict):
                results.append({"value": value, "error": raw_result})
                continue
            result.pop("scope", None)
            results.append({"value": value, **result})
    finally:
        file_path.write_text(original)

    payload: dict = {
        "scope": scope,
        "results": results,
        "source_restored": True,
    }
    best = _best_observed(results)
    if best:
        payload["best_observed"] = best
    return json.dumps(payload, separators=(",", ":"))


def _best_observed(results: list[dict]) -> dict | None:
    passing = [row for row in results if row.get("status") == "ALL_PASSED"]

    speedups = [
        (row, row.get("geomean", {}).get("speedup_factor"))
        for row in passing
    ]
    speedups = [(row, score) for row, score in speedups if score is not None]
    if speedups:
        row, score = max(speedups, key=lambda item: item[1])
        return {"value": row["value"], "geomean_speedup_factor": score}

    latencies = [
        (row, row.get("geomean", {}).get("latency_ms"))
        for row in passing
    ]
    latencies = [(row, score) for row, score in latencies if score is not None]
    if latencies:
        row, score = min(latencies, key=lambda item: item[1])
        return {"value": row["value"], "geomean_latency_ms": score}

    return None
