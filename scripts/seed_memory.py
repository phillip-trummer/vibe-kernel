"""Initialize memory with an optional target and/or baseline experiment.

This is not required to create or run a workspace. Run it before starting an
experiment to pin a comparison target, record the current src/ as the baseline,
or both. It refuses to replace existing memory.

    uv run python scripts/seed_memory.py \
        --workspace .runs/mla-flash \
        --target data/flashinfer-trace/solutions/vibe-kernel/opus4.8-25-07_flashinfer.json \
        --baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _resolve_target(raw: Path) -> Path:
    path = raw if raw.is_absolute() else Path.cwd() / raw
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"Error: target solution not found: {path}")
    return path


def _prepare_target(
    path: Path,
    definition: str,
    adapter: str,
    hardware: str,
) -> dict:
    solution = json.loads(path.read_text())
    if solution.get("definition") != definition:
        raise SystemExit(f"Error: {path} targets another task.")
    spec = solution["spec"]

    if adapter == "flashinfer" and "language" not in spec:
        languages = spec.pop("languages", [])
        mapping = {"cuda_cpp": "cuda", "pytorch": "python", "triton": "triton"}
        if len(languages) != 1 or languages[0] not in mapping:
            raise SystemExit(f"Error: cannot translate {path.name} to FlashInfer.")
        spec["language"] = mapping[languages[0]]
    elif adapter == "sol" and "languages" not in spec:
        language = spec.pop("language", None)
        mapping = {
            "cuda": "cuda_cpp",
            "cpp": "cuda_cpp",
            "python": "pytorch",
            "triton": "triton",
        }
        if language not in mapping:
            raise SystemExit(f"Error: cannot translate {path.name} to SOL.")
        spec["languages"] = [mapping[language]]
    spec["target_hardware"] = [hardware]
    return solution


def seed_memory(
    workspace: Path,
    target: Path | None = None,
    label: str | None = None,
    baseline: bool = False,
) -> None:
    from kernel_tools.tools import _tree
    from kernel_tools.tools._workspace import read_benchmark_manifest

    if target is None and not baseline:
        raise SystemExit("Error: choose --target, --baseline, or both.")
    if label is not None and target is None:
        raise SystemExit("Error: --label requires --target.")

    workspace = workspace.resolve()
    memory_path = workspace / _tree.MEMORY_PATH
    if memory_path.exists():
        raise SystemExit(
            f"Error: {memory_path} already exists; seed memory before starting "
            "the run."
        )

    try:
        manifest = read_benchmark_manifest(workspace)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: could not read workspace task: {exc}") from exc

    memory = None
    target_evaluation = None
    if target is not None:
        from kernel_tools.tools._benchmark import get_adapter
        from kernel_tools.tools._evaluation import aggregate

        try:
            memory = _tree.create_memory(workspace)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Error: could not read workspace task: {exc}") from exc
        target = _resolve_target(target)
        adapter = get_adapter(workspace, manifest)
        print(f"Benchmarking target: {target}")
        target_solution = _prepare_target(
            target,
            memory["task"],
            manifest["adapter"],
            (manifest["build_spec"].get("target_hardware") or ["LOCAL"])[0],
        )
        with tempfile.TemporaryDirectory(prefix="vibe-kernel-target-") as directory:
            prepared_target = Path(directory) / target.name
            prepared_target.write_text(json.dumps(target_solution))
            results = adapter.benchmark_target(prepared_target)
        target_evaluation = aggregate(results)
        memory["target"] = {
            "label": label or target.stem,
            "evaluation": target_evaluation.model_dump(mode="json"),
        }

    baseline_payload = None
    if baseline:
        from kernel_tools.tools.benchmark_kernel import benchmark_kernel

        print("Benchmarking baseline from workspace src/.")
        result = benchmark_kernel(workspace, scope="full")
        try:
            baseline_payload = json.loads(result)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error: baseline benchmark failed: {result}") from exc

    if memory is not None:
        _tree.save_memory(workspace, memory)

    if target_evaluation is not None:
        print(
            f"Pinned target {label or target.stem!r} "
            f"({target_evaluation.status}, "
            f"{target_evaluation.workload_count} workloads)."
        )

    if baseline_payload is not None:
        from kernel_tools.tools.create_branch import create_branch

        result = create_branch(
            workspace,
            slug="baseline",
            structure="Initial workspace implementation.",
            variant="Starting source from workspace setup.",
        )
        if result.startswith("Error:"):
            raise SystemExit(result)
        workloads = baseline_payload.get("workloads", {})
        print(
            "Seeded baseline as e0_baseline on b0_baseline "
            f"({baseline_payload.get('status')}, "
            f"{workloads.get('total', 0)} workloads)."
        )


def seed_target(workspace: Path, target: Path, label: str | None = None) -> None:
    seed_memory(workspace, target=target, label=label)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--label")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Fully benchmark and record the current src/ as e0_baseline.",
    )
    args = parser.parse_args(argv)
    seed_memory(args.workspace, args.target, args.label, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
