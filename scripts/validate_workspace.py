"""Validate a workspace without running kernels or changing files.

    python scripts/validate_workspace.py --workspace .runs/mla-sol
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _workload_ids(path: Path) -> set[str]:
    workload_ids = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            workload_id = json.loads(line)["workload"]["uuid"]
            if not isinstance(workload_id, str) or not workload_id:
                raise TypeError("uuid must be a non-empty string")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{path}:{line_number} has an invalid workload UUID: {exc}"
            ) from exc
        workload_ids.add(workload_id)
    return workload_ids


def validate_workspace(path: Path) -> dict:
    from kernel_tools.tools._benchmark import get_adapter
    from kernel_tools.tools._workspace import (
        read_benchmark_manifest,
        read_src_files,
    )
    from kernel_tools.workspace import resolve_workspace

    workspace = resolve_workspace(path)
    task_dir = workspace / "task"
    missing = [
        name
        for name in ("definition.json", "workloads.jsonl", "benchmark.json")
        if not (task_dir / name).is_file()
    ]
    if missing:
        raise ValueError(f"task is missing: {', '.join(missing)}")

    src_dir = workspace / "src"
    nested = [
        str(source.relative_to(src_dir))
        for source in src_dir.rglob("*")
        if source.is_file() and source.parent != src_dir
    ]
    if nested:
        raise ValueError(f"src must be flat; nested files: {', '.join(nested)}")

    sources = read_src_files(workspace)
    if not sources:
        raise ValueError("src has no source files")

    manifest = read_benchmark_manifest(workspace)
    workload_ids = _workload_ids(task_dir / "workloads.jsonl")
    unknown = [
        f"{label}={workload_id}"
        for label, workload_id in manifest["representative_workloads"].items()
        if workload_id not in workload_ids
    ]
    if unknown:
        raise ValueError(
            "representative workload UUIDs not found in task/workloads.jsonl: "
            + ", ".join(unknown)
        )

    adapter = get_adapter(workspace, manifest)
    task_spec = adapter.task_spec()
    adapter.build_contract()
    adapter.representative_axes()

    return {
        "workspace": workspace,
        "task": task_spec.model_dump(mode="json")["name"],
        "adapter": manifest["adapter"],
        "source_count": len(sources),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root; defaults to the current directory.",
    )
    args = parser.parse_args(argv)

    try:
        result = validate_workspace(args.workspace)
    except Exception as exc:
        print(f"Invalid workspace: {exc}", file=sys.stderr)
        return 1

    print(f"Workspace valid: {result['workspace']}")
    print(f"  task: {result['task']}")
    print(f"  adapter: {result['adapter']}")
    print(f"  sources: {result['source_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
