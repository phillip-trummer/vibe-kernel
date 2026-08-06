"""Workspace filesystem ABI.

Owns the portable workspace inputs (``task/benchmark.json`` and ``src/``), the
generic filesystem helpers the rest of the codebase composes against, and the
disposable ``.state/`` benchmark cache. No benchmark-framework or experiment
tree imports live here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ._evaluation import Evaluation
from ._workloads import REPRESENTATIVE_WORKLOAD_LABELS


# --- Paths ---
# These constants are workspace-relative names. Callers must join them to the
# explicit workspace path supplied by the MCP runtime.
TASK_DIR = Path("task")
ARCHIVE_DIR = Path("archive")
SRC_DIR = Path("src")
EXPERIMENTS_DIR = Path("experiments")
BENCHMARK_CACHE_PATH = Path(".state/benchmark_cache.json")
# Portable, user-owned benchmark contract. It selects the adapter and carries
# that backend's exact native build spec. Runtime tools read it directly; setup
# is only one way to create it.
BENCHMARK_MANIFEST_PATH = TASK_DIR / "benchmark.json"


# --- Experiment snapshots ---
def resolve_experiment_dir(workspace: Path, experiment_id: str) -> Path | str:
    """Resolve experiments/<experiment_id>/ with path-traversal protection.
    Returns the resolved Path on success, or an error message string."""
    experiments_dir = (workspace / EXPERIMENTS_DIR).resolve()
    exp_dir = (experiments_dir / experiment_id).resolve()
    if experiments_dir not in exp_dir.parents:
        return f"{experiment_id!r} is not a valid experiment id."
    if not exp_dir.is_dir():
        available = (
            sorted(p.name for p in experiments_dir.iterdir() if p.is_dir())
            if experiments_dir.is_dir()
            else []
        )
        return f"experiment {experiment_id!r} not found. Available: {available}"
    return exp_dir


def restore_experiment(workspace: Path, experiment_id: str) -> int | str:
    """Mirror a logged experiment into the working kernel."""
    exp_dir = resolve_experiment_dir(workspace, experiment_id)
    if not isinstance(exp_dir, Path):
        return exp_dir
    exp_files = [p for p in exp_dir.iterdir() if p.is_file()]
    if not exp_files:
        return f"experiment {experiment_id!r} has no source files; refusing to wipe working kernel."

    # Prepare working directory
    src_dir = (workspace / SRC_DIR).resolve()
    src_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale files
    exp_names = {p.name for p in exp_files}
    for stale in src_dir.iterdir():
        if stale.is_file() and stale.name not in exp_names:
            stale.unlink()

    # Restore snapshot
    for source in exp_files:
        shutil.copyfile(source, src_dir / source.name)
    return len(exp_files)


# --- Working source (src/) ---
def read_src_files(workspace: Path) -> list[tuple[str, str]]:
    """Return [(name, content), ...] for every file in src/, sorted by name.
    src/ is a flat directory of bare filenames (setup rejects nested source
    paths), so basenames are unambiguous keys."""
    src_dir = workspace / SRC_DIR
    if not src_dir.is_dir():
        return []
    return [
        (p.name, p.read_text())
        for p in sorted(src_dir.iterdir())
        if p.is_file()
    ]


def src_files_hash(file_pairs: list[tuple[str, str]]) -> str:
    """SHA256 over (name, content) pairs."""
    h = hashlib.sha256()
    for name, content in file_pairs:
        h.update(name.encode())
        h.update(b"\0")
        h.update(content.encode())
        h.update(b"\0")
    return h.hexdigest()


def solution_name_from_src_files(file_pairs: list[tuple[str, str]]) -> str:
    """The canonical `candidate_<hash16>` name identifying a src/ snapshot.
    Used as both the flashinfer Solution.name and the experiment dedup key."""
    return f"candidate_{src_files_hash(file_pairs)[:16]}"


# --- Portable benchmark manifest (task/benchmark.json) ---
def read_benchmark_manifest(workspace: Path) -> dict:
    """Read and validate the workspace's portable benchmark contract.

    ``build_spec`` deliberately stays an opaque dictionary at this layer. The
    selected adapter validates it with its backend-native model when it builds a
    solution, so one backend can expose options another backend does not have.
    """
    manifest_path = workspace / BENCHMARK_MANIFEST_PATH
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{manifest_path} not found; add the workspace benchmark "
            "manifest with 'adapter' and the backend-native 'build_spec'"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{manifest_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    adapter = manifest.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        raise ValueError(
            f"{manifest_path} must contain a non-empty string 'adapter'"
        )
    if not isinstance(manifest.get("hardware"), str) or not manifest["hardware"]:
        raise ValueError(
            f"{manifest_path} must contain a non-empty string 'hardware'"
        )
    if not isinstance(manifest.get("build_spec"), dict):
        raise ValueError(
            f"{manifest_path} must contain an object 'build_spec'"
        )
    reference_timing = manifest.get("reference_timing", True)
    if not isinstance(reference_timing, bool):
        raise ValueError(
            f"{manifest_path} 'reference_timing' must be a boolean"
        )
    # Backward-compatible default for workspaces created before this field was
    # introduced. Newly seeded manifests always persist the choice explicitly.
    manifest["reference_timing"] = reference_timing
    representatives = manifest.get("representative_workloads")
    expected_labels = set(REPRESENTATIVE_WORKLOAD_LABELS)
    if (
        not isinstance(representatives, dict)
        or set(representatives) != expected_labels
        or any(
            not isinstance(workload_id, str)
            or not workload_id
            for workload_id in representatives.values()
        )
        or len(set(representatives.values())) != len(REPRESENTATIVE_WORKLOAD_LABELS)
    ):
        raise ValueError(
            f"{manifest_path} 'representative_workloads' must map exactly "
            f"{', '.join(REPRESENTATIVE_WORKLOAD_LABELS)} to four unique "
            "workload UUIDs"
        )
    return manifest


def write_benchmark_manifest(workspace: Path, manifest: dict) -> None:
    """Write the authoritative workspace benchmark contract under ``task/``."""
    manifest_path = workspace / BENCHMARK_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Validate what was persisted so setup fails at the ownership boundary.
    read_benchmark_manifest(workspace)


def read_build_spec(workspace: Path) -> dict:
    """Return the manifest's backend-native build spec.

    Kept as a small accessor because adapters should not need to know the
    workspace layout. A missing manifest is an invalid runtime workspace and
    therefore raises a clear error instead of silently choosing defaults.
    """
    return read_benchmark_manifest(workspace)["build_spec"]


# --- Run benchmark cache (.state/benchmark_cache.json) ---
class BenchmarkCache(BaseModel):
    """Benchmark results for every source snapshot benchmarked in the current
    run, keyed by its candidate_<hash16> name. benchmark_kernel records into it;
    The experiment-recording tools recompute the current src hash and look it up,
    so a revert to any previously-benchmarked snapshot can be recorded without
    re-running (a single most-recent slot would lose the result the moment the
    agent explored past a peak and reverted). `last` names the most recently
    recorded snapshot."""
    entries: dict[str, Evaluation] = Field(default_factory=dict)
    last: str | None = None

    @classmethod
    def load(cls, path: Path) -> "BenchmarkCache":
        if path.is_file():
            return cls.model_validate_json(path.read_text())
        return cls()

    def record(self, solution_name: str, evaluation: Evaluation) -> None:
        self.entries[solution_name] = evaluation
        self.last = solution_name

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
