"""Translate a SOL-ExecBench Solution JSON into a FlashInfer Solution JSON.

    uv run python scripts/solution_to_flashinfer.py path/to/sol_solution.json \
        path/to/flashinfer_solution.json

The converter changes only the backend-specific build spec. Hardware is kept
as-is; workspace setup replaces it with the local CUDA device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LANGUAGES = {
    "cuda_cpp": "cuda",
    "pytorch": "python",
    "triton": "triton",
}


def _translate_spec(spec: dict) -> dict:
    translated = dict(spec)
    languages = translated.pop("languages", None)

    if languages is not None:
        if "language" in translated:
            raise SystemExit("Error: spec declares both language and languages.")
        if (
            not isinstance(languages, list)
            or len(languages) != 1
            or languages[0] not in LANGUAGES
        ):
            raise SystemExit(
                f"Error: FlashInfer has no builder for languages {languages!r}."
            )
        translated["language"] = LANGUAGES[languages[0]]
    elif not isinstance(translated.get("language"), str):
        raise SystemExit("Error: solution spec has no language.")

    # compile_options belongs to SOL's build spec.
    translated.pop("compile_options", None)
    translated.setdefault("destination_passing_style", False)
    return translated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)

    if not args.input_json.is_file():
        raise SystemExit(f"Error: {args.input_json} is not a file.")
    try:
        solution = json.loads(args.input_json.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Error: {args.input_json} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(solution, dict) or not isinstance(solution.get("spec"), dict):
        raise SystemExit(
            f"Error: {args.input_json} must contain a solution object with a spec."
        )

    solution["spec"] = _translate_spec(solution["spec"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(solution, indent=2) + "\n")

    name = solution.get("name", args.input_json.name)
    print(f"wrote FlashInfer solution {name!r} to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
