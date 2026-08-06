"""Create task/ and src/ for one kernel workspace.

    uv run python scripts/seed_task.py \
        --workspace .runs/mla-sol \
        --task mla_paged_decode_h16_ckv512_kpe64_ps1 \
        --adapter sol \
        --stub cuda

The command does not create .state/ or keep copies of baseline solutions.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

if __package__:
    from .seed_stub import STUB_LANGUAGES, make_stub_solution
else:
    from seed_stub import STUB_LANGUAGES, make_stub_solution


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "flashinfer-trace"
REPRESENTATIVE_LABELS = ("small", "medium", "large", "xlarge")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Error: could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Error: {path} must contain a JSON object.")
    return value


def _definitions(data_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted((data_dir / "definitions").glob("*/*.json"))
    }


def _resolve_solution(data_dir: Path, definition: str, name: str) -> Path:
    matches = []
    for path in sorted((data_dir / "solutions").rglob("*.json")):
        solution = _read_json(path)
        if solution.get("definition") != definition:
            continue
        if path.stem == name or name == solution.get("name"):
            return path
        if name in path.stem:
            matches.append(path)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.stem for path in matches)
        raise SystemExit(f"Error: {name!r} matches several solutions: {names}")
    raise SystemExit(
        f"Error: no solution matching {name!r} for task {definition!r}."
    )


def _load_workloads(path: Path) -> tuple[list[dict], list[str]]:
    workloads = []
    workload_ids = set()
    blobs = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            workload = json.loads(line)["workload"]
            workload_id = workload["uuid"]
            axes = workload["axes"]
            inputs = workload.get("inputs") or {}
            if not isinstance(workload_id, str) or not workload_id:
                raise TypeError("uuid must be a non-empty string")
            if workload_id in workload_ids:
                raise TypeError(f"duplicate uuid {workload_id!r}")
            if not isinstance(axes, dict) or any(
                type(size) is not int for size in axes.values()
            ):
                raise TypeError("axes must map names to integer sizes")
            if not isinstance(inputs, dict):
                raise TypeError("inputs must be an object")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SystemExit(
                f"Error: invalid workload at {path}:{line_number}: {exc}"
            ) from exc

        workload_ids.add(workload_id)
        workloads.append(workload)
        for spec in inputs.values():
            if spec.get("type") != "safetensors":
                continue
            raw = spec["path"]
            relative = Path(raw[2:] if raw.startswith("./") else raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"Error: invalid blob path {raw!r}.")
            blobs.add(relative.as_posix())

    if not workloads:
        raise SystemExit(f"Error: {path} contains no workloads.")
    return workloads, sorted(blobs)


def _auto_representatives(workloads: list[dict]) -> dict[str, str]:
    if len(workloads) < len(REPRESENTATIVE_LABELS):
        raise SystemExit(
            "Error: automatic representative selection requires at least "
            f"{len(REPRESENTATIVE_LABELS)} workloads."
        )

    ranked = sorted(
        workloads,
        key=lambda workload: math.prod(workload["axes"].values()),
    )
    selected = {
        label: ranked[
            (index + 1) * len(ranked) // len(REPRESENTATIVE_LABELS) - 1
        ]
        for index, label in enumerate(REPRESENTATIVE_LABELS)
    }

    print(
        "Auto-selected representatives by axis-product size "
        "(largest workload in each of four contiguous strata):"
    )
    for label, workload in selected.items():
        print(
            f"  {label}: uuid={workload['uuid']}, "
            f"axes={json.dumps(workload['axes'], sort_keys=True)}"
        )
    return {
        label: workload["uuid"]
        for label, workload in selected.items()
    }


def _load_solution(path: Path, definition: str, adapter: str) -> dict:
    solution = _read_json(path)
    if solution.get("definition") != definition:
        raise SystemExit(f"Error: {path} targets another task.")

    spec = solution.get("spec")
    expected = "language" if adapter == "flashinfer" else "languages"
    if not isinstance(spec, dict) or expected not in spec:
        raise SystemExit(
            f"Error: {path.name} is not a native {adapter} solution."
        )

    sources = solution.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SystemExit(f"Error: {path} has no source files.")
    for source in sources:
        name = source.get("path") if isinstance(source, dict) else None
        content = source.get("content") if isinstance(source, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or not isinstance(content, str)
        ):
            raise SystemExit(
                f"Error: {path} must contain text sources with plain filenames."
            )
    return solution


def _local_hardware() -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Error: PyTorch is required to detect the local CUDA device."
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("Error: no local CUDA device is available.")
    return torch.cuda.get_device_name(0)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def seed_workspace(
    *,
    workspace: Path,
    definition: str,
    adapter: str,
    data_dir: Path = DATA_DIR,
    baseline: str | None = None,
    stub: str | None = None,
    representative_workloads: list[str] | None = None,
    auto_representative_workloads: bool = False,
    reference_timing: bool = True,
    force: bool = False,
) -> None:
    workspace = workspace.resolve()
    data_dir = data_dir.resolve()
    if workspace == REPO_ROOT or workspace.parent == workspace:
        raise SystemExit("Error: choose a dedicated workspace directory.")
    if baseline is None and stub is None:
        raise SystemExit("Error: choose --baseline or --stub.")
    if baseline is not None and stub is not None:
        raise SystemExit("Error: --baseline and --stub are mutually exclusive.")
    if representative_workloads is not None and auto_representative_workloads:
        raise SystemExit(
            "Error: choose explicit representative workloads or automatic "
            "selection, not both."
        )
    if representative_workloads is None and not auto_representative_workloads:
        raise SystemExit(
            "Error: provide --representative-workloads SMALL MEDIUM LARGE "
            "XLARGE or pass --auto-representative-workloads."
        )

    definition_path = _definitions(data_dir).get(definition)
    if definition_path is None:
        raise SystemExit(
            f"Error: task {definition!r} not found. "
            "Run scripts/seed_task.py --list."
        )
    family = definition_path.parent.name
    workloads_path = data_dir / "workloads" / family / f"{definition}.jsonl"
    if not workloads_path.is_file():
        raise SystemExit(f"Error: workloads not found: {workloads_path}")

    workloads, blobs = _load_workloads(workloads_path)
    workload_ids = {workload["uuid"] for workload in workloads}
    missing = [blob for blob in blobs if not (data_dir / blob).is_file()]
    if missing:
        raise SystemExit(
            f"Error: {len(missing)} input blob(s) are missing. "
            "Run scripts/download_data.py without --metadata-only."
        )

    if auto_representative_workloads:
        representatives = _auto_representatives(workloads)
    else:
        if (
            len(set(representative_workloads)) != 4
            or any(item not in workload_ids for item in representative_workloads)
        ):
            raise SystemExit(
                "Error: representative workloads must be four unique task UUIDs."
            )
        representatives = dict(
            zip(REPRESENTATIVE_LABELS, representative_workloads)
        )

    definition_data = _read_json(definition_path)
    hardware = _local_hardware()
    build_hardware = "LOCAL" if adapter == "sol" else hardware.replace(" ", "_")
    if stub is not None:
        solution = make_stub_solution(
            definition_data,
            adapter,
            build_hardware,
            language=stub,
        )
        source_label = f"generated {stub} stub"
    else:
        solution_path = _resolve_solution(data_dir, definition, baseline or "")
        solution = _load_solution(solution_path, definition, adapter)
        source_label = solution_path.stem

    solution["spec"]["target_hardware"] = [build_hardware]

    generated = (workspace / "task", workspace / "src")
    if any(path.exists() or path.is_symlink() for path in generated) and not force:
        raise SystemExit(
            f"Error: {workspace} already contains task/ or src/. "
            "Resume it, choose another workspace, or pass --force to reset it."
        )
    if force:
        for name in (
            "task",
            "src",
            ".state",
            "archive",
            "experiments",
            "experiment_memory.md",
        ):
            _remove(workspace / name)

    task_dir = workspace / "task"
    src_dir = workspace / "src"
    task_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    _copy(definition_path, task_dir / "definition.json")
    _copy(workloads_path, task_dir / "workloads.jsonl")
    for blob in blobs:
        _copy(data_dir / blob, task_dir / blob)

    for source in solution["sources"]:
        (src_dir / source["path"]).write_text(source["content"])

    manifest = {
        "adapter": adapter,
        "hardware": hardware,
        "reference_timing": reference_timing,
        "build_spec": solution["spec"],
        "representative_workloads": representatives,
    }
    (task_dir / "benchmark.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Created workspace: {workspace}")
    print(f"  task: {definition} ({len(workloads)} workloads, {len(blobs)} blobs)")
    print(f"  adapter: {adapter}")
    print(f"  hardware: {hardware}")
    print(f"  reference timing: {reference_timing}")
    print(f"  starting source: {source_label}")
    languages = solution["spec"].get("languages")
    if not languages:
        language = solution["spec"].get("language")
        languages = [language] if language else ["unknown"]
    print(f"  language: {', '.join(languages)}")
    print(f"  sources: {len(solution['sources'])} file(s)")
    print("  state: not initialized")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--adapter", choices=("flashinfer", "sol"))
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--baseline")
    start.add_argument(
        "--stub",
        choices=STUB_LANGUAGES,
        metavar="LANGUAGE",
        help="Generate a cuda, python, or triton scaffold.",
    )
    representatives = parser.add_mutually_exclusive_group()
    representatives.add_argument(
        "--representative-workloads",
        nargs=4,
        metavar=("SMALL", "MEDIUM", "LARGE", "XLARGE"),
    )
    representatives.add_argument(
        "--auto-representative-workloads",
        action="store_true",
        help=(
            "Sort workloads by the product of their axes and select the largest "
            "item from each of four contiguous strata."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--reference-timing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Time the reference and report normalized speedup; use "
            "--no-reference-timing for correctness plus candidate latency only."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.list:
        for path in _definitions(args.data_dir.resolve()).values():
            print(path.stem)
        return 0

    missing = [
        name
        for name, value in (
            ("--workspace", args.workspace),
            ("--task", args.task),
            ("--adapter", args.adapter),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Error: required argument(s): {', '.join(missing)}")

    seed_workspace(
        workspace=args.workspace,
        definition=args.task,
        adapter=args.adapter,
        data_dir=args.data_dir,
        baseline=args.baseline,
        stub=args.stub,
        representative_workloads=args.representative_workloads,
        auto_representative_workloads=args.auto_representative_workloads,
        reference_timing=args.reference_timing,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
