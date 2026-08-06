"""flashinfer-bench benchmark adapter.

The only module that imports `flashinfer_bench.*`. Implements the
BenchmarkAdapter interface (see _benchmark.py) for flashinfer: `FlashInferAdapter`
owns the task fixtures (definition + workloads) so the harness never handles
native flashinfer types (`Definition`, `Workload`, `Trace`, `Solution`,
`EvaluationStatus`) — they stay inside this file, and only neutral results
(WorkloadResult leaves, TaskSpec, runnable+inputs) cross the boundary. A future
swap of flashinfer-bench is scoped to this file plus a sibling adapter.

flashinfer imports are deferred to function bodies so that importing the tools
does not pull torch + flashinfer eagerly.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from ._torch_build_log import denoise
from ._torch_build_log import strip_build_noise as _strip_build_noise
from .._workspace import (
    ARCHIVE_DIR,
    TASK_DIR,
    read_benchmark_manifest,
    read_build_spec,
    read_src_files,
    solution_name_from_src_files,
)
from .._workloads import (
    representative_item_for_label,
    select_representative_workloads,
)
from .._evaluation import (
    AxisField,
    Correctness,
    TaskSpec,
    TensorField,
    Tolerance,
    WorkloadResult,
    normalize_outcome,
)


# --- Build-spec defaults (Solution construction) ---
# flashinfer entry_point is "<file>::<symbol>". The host entry file follows the
# kernel's language: cuda compiles main.cpp (kernel.cu/.h are included); python
# and triton import main.py. The symbol (run) is the same across languages.
ENTRY_FILE_BY_LANGUAGE = {"cuda": "main.cpp", "python": "main.py", "triton": "main.py"}
ENTRY_SYMBOL = "run"
AUTHOR = "agent"


class FlashInferAdapter:
    """Loads the task fixtures once, then exposes only neutral results — native
    flashinfer types never leave this class. The kernel's source language comes
    from the baseline's frozen build spec, not config."""

    def __init__(
        self,
        workspace: Path,
        representative_workloads: dict[str, str],
    ) -> None:
        self.workspace = workspace
        self.representative_workloads = representative_workloads
        self.definition, self.workload_traces = _load_task(workspace)
        self.reference_timing = read_benchmark_manifest(workspace)[
            "reference_timing"
        ]
        self.benchmark_config = _bench_config(self.reference_timing)
        self.eval_config = self.benchmark_config.resolve_eval_config(self.definition)

    def benchmark(self, scope: str) -> list[WorkloadResult]:
        """Build the current src/ kernel, run it against the workloads (all, or
        the representative four when scope='smoke'), archive the run, and return
        one neutral WorkloadResult per workload. Shared harness code aggregates
        the leaves — this adapter never scores."""
        solution = self._build_solution()
        _append_solution_to_archive(self.workspace, solution)
        workloads = self.workload_traces
        if scope == "smoke":
            workloads, _ = select_representative_workloads(
                workloads,
                self.representative_workloads,
                lambda trace: str(trace.workload.uuid),
            )
        traces = _run_benchmark(
            self.workspace,
            self.definition,
            solution,
            workloads,
            self.benchmark_config,
        )
        _append_traces_to_archive(self.workspace, traces)
        return _workload_results(
            traces,
            self.representative_workloads,
            self.eval_config,
        )

    def benchmark_target(self, target_path: Path) -> list[WorkloadResult]:
        """Benchmark the target Solution at target_path (a .json file) against the
        full workload suite. It is benchmarked with its own build spec, so it
        may be a different language/runtime than the agent's kernel."""
        solution = self._load_solution_file(target_path)
        traces = _run_benchmark(
            self.workspace,
            self.definition,
            solution,
            self.workload_traces,
            self.benchmark_config,
        )
        return _workload_results(
            traces,
            self.representative_workloads,
            self.eval_config,
        )

    def prepare_baseline(
        self,
        baseline_path: Path,
        build_spec_overrides: dict | None = None,
    ) -> tuple[list[tuple[str, str]], dict]:
        """Load a baseline and return its sources plus resolved native spec."""
        solution = self._load_solution_file(baseline_path)
        files = [(s.path, s.content) for s in solution.sources]
        spec = _resolve_build_spec(
            solution.spec.model_dump(mode="json"),
            build_spec_overrides or {},
        )
        # Validate the complete source/spec pair through the backend's own types
        # before setup makes it the workspace contract.
        _solution_with_spec(self.definition.name, files, spec)
        return files, spec

    def _load_solution_file(self, path: Path):
        """Load the flashinfer Solution at `path` and validate it targets this
        task."""
        from flashinfer_bench.data import Solution

        try:
            solution = Solution.model_validate_json(path.read_text())
        except Exception as e:
            raise ValueError(f"{path.name} is not a valid solution file: {e}")
        if solution.definition != self.definition.name:
            raise ValueError(
                f"solution targets definition {solution.definition!r}, but "
                f"the task definition is {self.definition.name!r}"
            )
        return solution

    def representative_axes(self) -> dict[str, dict[str, int]]:
        """Map each representative label (small/medium/large/xlarge) to its
        workload's `axes` — the concrete axis-name -> integer shape declared by
        the task fixtures. Labels that collapse onto the same workload are
        deduped, so the result may have fewer than four entries."""
        selected, labels = select_representative_workloads(
            self.workload_traces,
            self.representative_workloads,
            lambda trace: str(trace.workload.uuid),
        )
        return {label: dict(t.workload.axes) for label, t in zip(labels, selected)}

    def prepare_reference_baseline(
        self,
        build_spec_overrides: dict | None = None,
    ) -> tuple[list[tuple[str, str]], dict] | None:
        """Package the task's reference and its resolved native build spec."""
        import torch
        from flashinfer_bench.data import BuildSpec, SupportedBindings

        reference = (getattr(self.definition, "reference", "") or "").strip()
        if not reference:
            return None
        content = reference.replace("\r\n", "\n").replace("\r", "\n") + "\n"
        entry_file = ENTRY_FILE_BY_LANGUAGE["python"]
        base_spec = BuildSpec(
            language="python",
            target_hardware=[torch.cuda.get_device_name(0).replace(" ", "_")],
            entry_point=f"{entry_file}::{ENTRY_SYMBOL}",
            binding=SupportedBindings.TORCH,
            destination_passing_style=False,
        )
        files = [(entry_file, content)]
        spec = _resolve_build_spec(
            base_spec.model_dump(mode="json"),
            build_spec_overrides or {},
        )
        _solution_with_spec(self.definition.name, files, spec)
        return files, spec

    def build_contract(self) -> str | None:
        """One agent-facing sentence stating the fixed build contract the working
        kernel must honor — entry symbol, calling convention (value-return vs.
        out-params), and any available build dependencies — derived from the
        manifest's build spec."""
        spec = read_build_spec(self.workspace)
        return _build_contract_text(spec, self.definition) if spec else None

    def task_spec(self) -> TaskSpec:
        """Map the flashinfer Definition into the neutral TaskSpec, including its
        correctness bar after FlashInfer resolves bundled op/definition policy."""
        d = self.definition.model_dump(mode="json")
        cfg = self.eval_config
        return TaskSpec(
            name=d["name"],
            description=d.get("description") or "",
            op_type=d.get("op_type"),
            axes={k: _axis_field(v) for k, v in (d.get("axes") or {}).items()},
            inputs={k: _tensor_field(v) for k, v in (d.get("inputs") or {}).items()},
            outputs={k: _tensor_field(v) for k, v in (d.get("outputs") or {}).items()},
            reference=d.get("reference") or "",
            constraints=d.get("constraints") or [],
            tolerance=Tolerance(
                max_atol=cfg.atol,
                max_rtol=cfg.rtol,
                required_matched_ratio=getattr(cfg, "required_matched_ratio", None),
            ),
        )

    def strip_build_noise(self, text: str) -> str:
        """Drop this backend's build-system chatter (ninja steps, torch
        cpp_extension, builder banners) from externally captured output — e.g.
        ncu's — so a profiler's report dominates. Other lines pass through."""
        return _strip_build_noise(text)

    def prewarm(self) -> None:
        """Compile the current src/ kernel to its on-disk artifact so a separate
        ncu child process reuses it instead of recompiling under instrumentation."""
        _prewarm_build(self.definition, self._build_solution())

    def build_profilable(self, label: str) -> tuple[Callable, list]:
        """Build the current src/ kernel and materialize one representative
        workload's inputs. Returns (runnable, inputs); call runnable(*inputs)."""
        workload = representative_item_for_label(
            self.workload_traces,
            label,
            self.representative_workloads,
            lambda trace: str(trace.workload.uuid),
        ).workload
        runnable = _build_runnable(self.definition, self._build_solution())
        inputs = _materialize_inputs(self.workspace, self.definition, workload)
        return runnable, inputs

    def _build_solution(self):
        return _build_solution_from_src(self.workspace, self.definition.name)


def _build_contract_text(spec: dict, definition) -> str:
    """Compose the build-contract sentence from a frozen build spec dict: the
    entry symbol, value-return vs. out-param calling convention, and any
    available build dependencies."""
    entry = spec.get("entry_point", "")
    symbol = entry.split("::", 1)[1] if "::" in entry else entry
    outputs = list(getattr(definition, "outputs", {}) or {})
    out_str = ", ".join(outputs) if outputs else "its outputs"
    if spec.get("destination_passing_style", True):
        convention = f"writes {out_str} into caller-provided out-parameters"
    else:
        convention = f"returns ({out_str}) by value"
    parts = [f"The working kernel must define `{symbol}`, which {convention}."]
    deps = spec.get("dependencies") or []
    if deps:
        parts.append(f"Available build dependencies: {', '.join(deps)}.")
    return " ".join(parts)


# --- Task fixtures ---
def _load_task(workspace: Path) -> tuple[object, list]:
    """Parse task/definition.json + task/workloads.jsonl from the workspace."""
    from flashinfer_bench.data import Definition, Trace

    task_dir = workspace / TASK_DIR
    def_path = task_dir / "definition.json"
    wl_path = task_dir / "workloads.jsonl"

    definition = Definition.model_validate_json(def_path.read_text())
    workloads = [
        Trace.model_validate_json(line)
        for line in wl_path.read_text().splitlines()
        if line.strip()
    ]
    return definition, workloads


# --- Solution construction ---
def _build_solution_from_src(workspace: Path, definition_name: str):
    """Build a Solution from ``src/`` and ``task/benchmark.json``."""
    files = read_src_files(workspace)
    if not files:
        raise RuntimeError("the working kernel has no source files")
    spec = read_build_spec(workspace)
    if spec is None:
        raise RuntimeError("the working kernel has no build contract")
    return _solution_with_spec(definition_name, files, spec)


def _solution_with_spec(definition_name: str, files: list[tuple[str, str]], spec: dict):
    """Build a flashinfer Solution from (name, content) source pairs and a frozen
    build spec dict (the baseline's own spec, governing the whole run)."""
    from flashinfer_bench.data import BuildSpec, Solution, SourceFile

    return Solution(
        name=solution_name_from_src_files(files),
        definition=definition_name,
        author=AUTHOR,
        spec=BuildSpec.model_validate(spec),
        sources=[SourceFile(path=name, content=content or "\n") for name, content in files],
    )


def _resolve_build_spec(spec: dict, overrides: dict) -> dict:
    """Apply options through FlashInfer's native BuildSpec schema."""
    from flashinfer_bench.data import BuildSpec

    return BuildSpec.model_validate(
        {**spec, **overrides},
        extra="forbid",
    ).model_dump(mode="json")


# --- Benchmark execution ---
def _bench_config(reference_timing: bool = True):
    """Load FlashInfer's bundled policy, overriding only reference timing."""
    from flashinfer_bench.bench import BenchmarkConfig
    return BenchmarkConfig.default(profile_baseline=reference_timing)


def _run_benchmark(
    workspace: Path,
    definition,
    solution,
    workloads: list,
    benchmark_config,
) -> list:
    """Run flashinfer-bench on `solution` against `workloads`. Returns the
    list of resulting Traces. The TraceSet is rooted at TASK_DIR so relative
    safetensors paths in workloads.jsonl (e.g. "./blob/...") resolve correctly.

    `workloads` is a list of Trace objects (definition+workload populated,
    solution/evaluation null) — this matches the flashinfer schema, where a
    standalone Workload is stored as a Trace and `TraceSet.workloads` is typed
    as `Dict[str, List[Trace]]`."""
    from flashinfer_bench.bench import Benchmark
    from flashinfer_bench.data import TraceSet

    trace_set = TraceSet(
        root=workspace / TASK_DIR,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
    )
    result_set = Benchmark(trace_set, benchmark_config).run_all(dump_traces=False)
    return result_set.traces.get(definition.name, [])


def _build_runnable(definition, solution):
    """Compile a Solution into a runnable kernel via flashinfer's BuilderRegistry."""
    from flashinfer_bench.compile import BuilderRegistry
    return BuilderRegistry.get_instance().build(definition, solution)


def _prewarm_build(definition, solution) -> None:
    """Compile `solution` into its on-disk build directory so a separate process
    (the profile_kernel ncu child) reuses the artifact instead of recompiling.
    The build dir is keyed by content hash, so the child derives the same path.

    Builds via TorchBuilder directly, not _build_runnable: the registry's
    in-process cache would short-circuit without guaranteeing the on-disk
    artifact exists (e.g. after a benchmark run cleaned the dir)."""
    from flashinfer_bench.compile.builders import TorchBuilder
    TorchBuilder().build(definition, solution)


def _materialize_inputs(
    workspace: Path,
    definition,
    workload,
    device: str = "cuda:0",
) -> list:
    """Materialize input values for a workload, loading safetensors if needed.

    Returns a positional list in definition.inputs order, matching flashinfer's
    own evaluators — call the runnable as ``runnable(*inputs)``."""
    from flashinfer_bench.bench.utils import gen_inputs, load_safetensors

    safe_tensors = (
        load_safetensors(definition, workload, workspace / TASK_DIR)
        if any(d.type == "safetensors" for d in workload.inputs.values())
        else None
    )
    return gen_inputs(definition, workload, device=device, safe_tensors=safe_tensors)


# --- Archive I/O (archive/solutions.jsonl + archive/traces.jsonl) ---
def _append_solution_to_archive(workspace: Path, solution) -> None:
    """Append `solution` to archive/solutions.jsonl, deduped by name."""
    sol_path = _archive_dir(workspace) / "solutions.jsonl"
    seen: set[str] = set()
    if sol_path.exists():
        for line in sol_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                continue
    if solution.name in seen:
        return
    with sol_path.open("a") as f:
        f.write(solution.model_dump_json() + "\n")


def _append_traces_to_archive(workspace: Path, traces) -> None:
    """Append `traces` to archive/traces.jsonl. No-op if traces is empty."""
    if not traces:
        return
    with (_archive_dir(workspace) / "traces.jsonl").open("a") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")


def _archive_dir(workspace: Path) -> Path:
    archive_dir = workspace / ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


# --- Neutral leaf mapping (one WorkloadResult per trace) ---
def _workload_results(
    traces, representative_workloads: dict[str, str], eval_config
) -> list[WorkloadResult]:
    """Map each flashinfer Trace to a neutral WorkloadResult leaf. Aggregation
    (geomean, representative pick, failure histogram) is shared harness code, not
    this adapter's job."""
    from flashinfer_bench.data import EvaluationStatus

    names_by_uuid = {
        workload_uuid: name
        for name, workload_uuid in representative_workloads.items()
    }
    return [
        _workload_result(
            trace,
            EvaluationStatus,
            eval_config,
            names_by_uuid.get(str(trace.workload.uuid)),
        )
        for trace in traces
    ]


def _workload_result(
    trace, EvaluationStatus, eval_config, representative_name: str | None = None
) -> WorkloadResult:
    passed = trace.evaluation.status == EvaluationStatus.PASSED
    perf = trace.evaluation.performance if passed else None
    has_reference_timing = bool(
        perf
        and perf.reference_latency_ms is not None
        and perf.reference_latency_ms > 0
        and perf.speedup_factor is not None
        and perf.speedup_factor > 0
    )
    return WorkloadResult(
        axes=dict(trace.workload.axes),
        outcome=normalize_outcome(trace.evaluation.status.value),
        latency_ms=round(perf.latency_ms, 6) if perf else None,
        reference_latency_ms=(
            round(perf.reference_latency_ms, 6)
            if has_reference_timing
            else None
        ),
        speedup_factor=(
            round(perf.speedup_factor, 4)
            if has_reference_timing
            else None
        ),
        tolerance=_tolerance(eval_config),
        correctness=_correctness(trace),
        diagnostic=None if passed else _workload_diagnostic(trace, eval_config),
        representative_name=representative_name,
    )


def _correctness(trace) -> Correctness | None:
    """Structured correctness from a trace's native record. Non-finite error
    metrics become flags (has_nan / has_inf) with the numeric field left null, so
    the leaf stays JSON-clean."""
    c = getattr(trace.evaluation, "correctness", None)
    if c is None:
        return None
    abs_err, rel_err = c.max_absolute_error, c.max_relative_error
    has_inf = any(isinstance(v, float) and math.isinf(v) for v in (abs_err, rel_err))
    has_nan = any(isinstance(v, float) and math.isnan(v) for v in (abs_err, rel_err))
    return Correctness(
        max_abs_error=_finite_metric(abs_err),
        max_rel_error=_finite_metric(rel_err),
        has_nan=has_nan,
        has_inf=has_inf,
    )


def _tolerance(cfg) -> Tolerance:
    return Tolerance(
        max_atol=cfg.atol,
        max_rtol=cfg.rtol,
        required_matched_ratio=getattr(cfg, "required_matched_ratio", None),
    )


def _finite_metric(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return _metric(value)


def _axis_field(a: dict) -> AxisField:
    """Map a flashinfer axis dump (AxisConst / AxisVar / AxisExpr) to AxisField."""
    return AxisField(
        kind=a.get("type", "var"),
        value=a.get("value"),
        expression=a.get("expression"),
        description=a.get("description"),
    )


def _tensor_field(t: dict) -> TensorField:
    """Map a flashinfer TensorSpec dump to the neutral TensorField."""
    return TensorField(
        shape=t.get("shape"),
        dtype=t.get("dtype", "?"),
        description=t.get("description"),
    )


# --- Diagnostics (backend-specific log/correctness normalization) ---
# All of this turns flashinfer's native build logs and correctness records into
# the normalized strings the contract carries, so the harness never parses them.
# The toolchain-level cleanup (ninja/cpp_extension noise, build paths) is shared;
# only what is flashinfer's own lives here.
# Normalize flashinfer's repeated-build wrapper.
_SKIPPED_RE = re.compile(r"^Solution skipped after \d+ failures?\. Last error:\s*")


def _workload_diagnostic(trace, eval_config) -> str | None:
    # Compose a correctness summary and a filtered log tail into one note.
    parts = []
    correctness = _correctness_text(
        getattr(trace.evaluation, "correctness", None),
        eval_config,
    )
    if correctness:
        parts.append(correctness)
    log = getattr(trace.evaluation, "log", "") or ""
    tail = _diagnostic_tail(log, trace.evaluation.status.value)
    if tail:
        parts.append(tail)
    return "\n".join(parts) or None


def _diagnostic_tail(text: str, status: str) -> str:
    """Unwrap flashinfer's repeated-build wrapper, then hand the log to the shared
    toolchain cleanup."""
    unwrapped = "\n".join(_SKIPPED_RE.sub("", line) for line in text.strip().splitlines())
    return denoise(unwrapped, compile_error=status == "COMPILE_ERROR")


def _correctness_text(correctness, cfg) -> str | None:
    if correctness is None:
        return None
    rel = correctness.max_relative_error
    abs_err = correctness.max_absolute_error
    # Non-finite metrics are sentinels for non-finite output.
    for value in (abs_err, rel):
        if isinstance(value, float) and math.isinf(value):
            return "output contains Inf"
        if isinstance(value, float) and math.isnan(value):
            return "output contains NaN"
    # Show each error against its tolerance and the gate: a point fails only
    # when it exceeds both, so the agent can tell a real bug (both exceeded)
    # from benign precision (one within tolerance).
    return (
        f"max relative error {_metric(rel)} (rtol {_metric(cfg.rtol)}), "
        f"max absolute error {_metric(abs_err)} (atol {_metric(cfg.atol)}); "
        f"a point fails only when it exceeds both tolerances"
    )


def _metric(value):
    return round(value, 6) if isinstance(value, float) else value
