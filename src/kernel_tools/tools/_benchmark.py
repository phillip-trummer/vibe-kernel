"""Portable benchmark-adapter interface — the product integration contract.

The harness talks to whatever benchmark framework runs kernels (flashinfer
today, SOL/kernelbench later) only through this interface, so tools never handle
a framework's native types. An adapter owns its task fixtures internally and
returns neutral results: a `TaskSpec`, and per-workload `WorkloadResult` leaves
that shared harness code (`aggregate`) folds into the stored `Evaluation` (see
_evaluation.py). No framework imports live here — adding a benchmark means adding
a module under `tools/adapters/` and registering its name in `get_adapter()`,
not touching the tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from ._evaluation import TaskSpec, WorkloadResult
from ._workspace import read_benchmark_manifest


class BenchmarkUnavailable(RuntimeError):
    """The benchmark could not evaluate the kernel, for reasons that are not the
    kernel's fault (no usable timer, a broken toolchain, staging failed).

    Distinct from a kernel that fails: those are `WorkloadResult`s with a failing
    outcome, which the agent should act on. This is the absence of a verdict, so
    an adapter raises it rather than inventing leaves — the message reaches the
    agent, and nothing is scored or logged. Keep the message in agent vocabulary."""


class BenchmarkAdapter(Protocol):
    """What every benchmark adapter exposes to the harness. Any method may raise
    `BenchmarkUnavailable` when the benchmark itself is unusable."""

    def benchmark(self, scope: str) -> list[WorkloadResult]:
        """Build the current working kernel, run it against the workloads ('full'
        suite or 'smoke' representatives), and return one neutral WorkloadResult
        per workload. The adapter does not aggregate — shared harness code folds
        the leaves into the stored Evaluation."""
        ...

    def benchmark_target(self, target_path: Path) -> list[WorkloadResult]:
        """Benchmark the comparison target the user supplied at target_path (in
        this adapter's native format) against the full workload suite; return
        neutral leaves. How the target is packaged there is the adapter's
        concern."""
        ...

    def representative_axes(self) -> dict[str, dict[str, int]]:
        """Map each representative label to its workload's concrete axes."""
        ...

    def task_spec(self) -> TaskSpec:
        """The neutral task description stored when memory is initialized."""
        ...

    def prepare_baseline(
        self,
        baseline_path: Path,
        build_spec_overrides: dict | None = None,
    ) -> tuple[list[tuple[str, str]], dict]:
        """The user's starting kernel at baseline_path (in this adapter's native
        format), returned as working-kernel source files plus the resolved native
        build spec. Setup persists the pair into ``src/`` and
        ``task/benchmark.json``; the adapter never writes workspace state."""
        ...

    def prepare_reference_baseline(
        self,
        build_spec_overrides: dict | None = None,
    ) -> tuple[list[tuple[str, str]], dict] | None:
        """The task's reference packaged as working-kernel source files (name,
        content) plus its resolved native build spec, or None if the task has no
        runnable reference. Setup uses this when [task] baseline = 'reference'."""
        ...

    def build_contract(self) -> str | None:
        """One agent-facing sentence stating the fixed build contract the working
        kernel must honor (entry symbol, calling convention, available
        dependencies), derived from the frozen build spec. None if the adapter
        has no build contract to surface."""
        ...

    def strip_build_noise(self, text: str) -> str:
        """Drop this adapter's build-system chatter from externally captured
        output (e.g. a profiler's), so its report dominates. Adapters with no
        build noise return text unchanged."""
        ...

    def prewarm(self) -> None:
        """Build the current working kernel to a reusable on-disk artifact."""
        ...

    def build_profilable(self, label: str) -> tuple[Callable, list]:
        """Build the current kernel and materialize one representative workload's
        complete call arguments. Destination-passing kernels include preallocated
        outputs after their inputs. Returns (runnable, arguments); call
        runnable(*arguments)."""
        ...


def get_adapter(
    workspace: Path,
    manifest: dict | None = None,
) -> BenchmarkAdapter:
    """Construct the adapter selected by ``task/benchmark.json``.

    Setup may pass an in-memory partial manifest before it has resolved the
    baseline build spec. Runtime callers always use the persisted manifest.
    """
    manifest = (
        manifest if manifest is not None else read_benchmark_manifest(workspace)
    )
    name = manifest.get("adapter")
    representative_workloads = manifest["representative_workloads"]
    if name == "flashinfer":
        from .adapters.flashinfer import FlashInferAdapter

        return FlashInferAdapter(workspace, representative_workloads)
    if name == "sol":
        from .adapters.sol import SOLAdapter

        return SOLAdapter(workspace, representative_workloads)
    raise ValueError(f"unknown benchmark adapter {name!r}")
