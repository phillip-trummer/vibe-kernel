"""Translate a FlashInfer Solution JSON into a SOL-ExecBench Solution JSON.

SOL-ExecBench uses a BuildSpec incompatible with flashinfer-bench: `languages`
(a list) replaces `language`, and `target_hardware` is narrowed to the enum
{B200, LOCAL}. Everything else about the solution object carries over unchanged.

    uv run python scripts/solution_to_sol.py path/to/flashinfer_solution.json \
        path/to/sol_solution.json

The converter is intentionally file-to-file so you can point it at any baseline
or target JSON and write the SOL-native result wherever you want.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# flashinfer's BuildSpec.language -> SOL's BuildSpec.languages. SOL folds C++ and
# CUDA into one token and calls plain torch "pytorch"; tilelang has no SOL builder.
LANGUAGES = {"cuda": "cuda_cpp", "cpp": "cuda_cpp", "python": "pytorch", "triton": "triton"}

# SOL accepts only these two. LOCAL makes its packager detect the compile machine's
# arch; B200 forces sm_100a, which will not run anywhere else.
HARDWARE = ("LOCAL", "B200")

# SOL's own default is 120s, which a real kernel.cu overruns, but this script
# only translates solution JSON and does not run benchmarks.


def _translate_spec(spec: dict, hardware: str) -> dict:
    """Rewrite a flashinfer BuildSpec into SOL's schema."""
    translated = dict(spec)

    if "languages" in translated and "language" not in translated:
        translated["target_hardware"] = [hardware]
    else:
        language = translated.pop("language", None)
        if language not in LANGUAGES:
            raise SystemExit(f"Error: SOL has no builder for language {language!r}.")
        translated.pop("target_hardware", None)
        translated["languages"] = [LANGUAGES[language]]
        translated["target_hardware"] = [hardware]

    # SOL defaults this to True and then calls fn(*inputs, *outputs), discarding the
    # return value — so an unset flag silently fails every workload. Pin it.
    translated.setdefault("destination_passing_style", False)
    return translated


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_json", type=Path, help="FlashInfer solution JSON to translate.")
    parser.add_argument("output_json", type=Path, help="SOL solution JSON to write.")
    parser.add_argument("--hardware", choices=HARDWARE, default="LOCAL", help="SOL target hardware (default: LOCAL).")
    args = parser.parse_args(argv)

    if not args.input_json.is_file():
        raise SystemExit(f"Error: {args.input_json} is not a file.")
    try:
        solution = json.loads(args.input_json.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: {args.input_json} is not valid JSON: {exc}") from exc
    if not isinstance(solution, dict) or "spec" not in solution or not isinstance(solution["spec"], dict):
        raise SystemExit(f"Error: {args.input_json} must contain a solution object with a spec.")

    solution["spec"] = _translate_spec(solution["spec"], args.hardware)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(solution, indent=2) + "\n")

    print(f"wrote SOL solution {solution.get('name', args.input_json.name)!r} to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
