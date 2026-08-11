"""SOL-ExecBench benchmark adapter.

The only module that imports `sol_execbench.*`. Implements the BenchmarkAdapter
interface (see _benchmark.py): `SOLAdapter` owns the task fixtures so the harness
never handles native SOL types (`Definition`, `Workload`, `Trace`, `Solution`,
`EvaluationStatus`) — only neutral results cross the boundary.

SOL is a sibling of flashinfer-bench, not a superset: `BuildSpec.languages` is a
list of different tokens (cuda -> cuda_cpp, python -> pytorch) and
`target_hardware` is the enum {B200, LOCAL}. So a solution native to one is not
native to the other; `scripts/solution_to_sol.py` translates ours out. The task
fixtures (definition.json, workloads.jsonl) do carry over unchanged.

SOL runs each evaluation in two subprocesses (compile, then an eval driver that
emits one Trace per workload on stdout). It resolves safetensors against
FLASHINFER_TRACE_DIR rather than staging them, so we point that at the task.

sol_execbench imports are deferred to function bodies so importing the tools
does not pull torch eagerly.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from .._benchmark import BenchmarkUnavailable
from ._torch_build_log import denoise
from ._torch_build_log import strip_build_noise as _strip_build_noise
from .._evaluation import (
    AxisField,
    Correctness,
    TaskSpec,
    TensorField,
    Tolerance,
    WorkloadResult,
    normalize_outcome,
)
from .._workloads import representative_item_for_label, select_representative_workloads
from .._workspace import (
    ARCHIVE_DIR,
    TASK_DIR,
    read_benchmark_manifest,
    read_build_spec,
    read_src_files,
    solution_name_from_src_files,
)


# --- Solution construction ---
AUTHOR = "agent"

# SOL's packager compiles these through torch's cpp_extension; the rest are imported.
CPP_LANGUAGES = frozenset({"cuda_cpp", "cutlass", "cudnn", "cublas"})

# Where prewarm() leaves the compiled artifact for the profiler's child process.
PROFILE_BUILD_DIR = Path(".state/profile_build")

# SOL's own defaults are 120s / 600s; a real kernel.cu overruns the first.
COMPILE_TIMEOUT_S = 1200
EVAL_TIMEOUT_S = 1200

# SOL reports an unusable timer with the same per-workload status (RUNTIME_ERROR)
# as a kernel that genuinely fails to run — only the message separates them. A
# broken timer is not the kernel's fault, so it becomes BenchmarkUnavailable
# rather than a failing leaf the agent would try to fix.
INFRA_LOG_PREFIXES = ("Timing failed:",)


class SOLAdapter:
    """Loads the task fixtures once, then exposes only neutral results — native
    SOL types never leave this class."""

    def __init__(
        self,
        workspace: Path,
        representative_workloads: dict[str, str],
    ) -> None:
        self.workspace = workspace
        self.representative_workloads = representative_workloads
        self.definition, self.workloads = _load_task(workspace)
        self.reference_timing = read_benchmark_manifest(workspace)[
            "reference_timing"
        ]

    def benchmark(self, scope: str) -> list[WorkloadResult]:
        """Build the current src/ kernel, run it against the workloads (all, or the
        representative four when scope='smoke'), archive the run, and return one
        neutral WorkloadResult per workload."""
        solution = self._build_solution()
        _append_solution_to_archive(self.workspace, solution)
        workloads = self.workloads
        if scope == "smoke":
            workloads, _ = select_representative_workloads(
                workloads,
                self.representative_workloads,
                lambda workload: str(workload.uuid),
            )
        results = _evaluate(
            self.workspace,
            self.definition,
            solution,
            workloads,
            reference_timing=self.reference_timing,
        )
        return self._mark_representatives(workloads, results)

    def benchmark_target(self, target_path: Path) -> list[WorkloadResult]:
        """Benchmark the target Solution at target_path against the full suite; it
        carries its own build spec, so it may be a different language than the
        agent's kernel."""
        solution = self._load_solution_file(target_path)
        results = _evaluate(
            self.workspace,
            self.definition,
            solution,
            self.workloads,
            reference_timing=self.reference_timing,
            archive=False,
        )
        return self._mark_representatives(self.workloads, results)

    def prepare_starting_kernel(
        self,
        starting_kernel_path: Path,
        build_spec_overrides: dict | None = None,
    ) -> tuple[list[tuple[str, str]], dict]:
        """Load a starting kernel and return its sources plus resolved native spec."""
        solution = self._load_solution_file(starting_kernel_path)
        files = [(s.path, s.content) for s in solution.sources]
        spec = _resolve_build_spec(
            solution.spec.model_dump(mode="json"),
            build_spec_overrides or {},
        )
        _solution_with_spec(self.definition.name, files, spec)
        return files, spec

    def _load_solution_file(self, path: Path):
        from sol_execbench import Solution

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
        """Map each representative label to its workload's concrete axes."""
        selected, labels = select_representative_workloads(
            self.workloads,
            self.representative_workloads,
            lambda workload: str(workload.uuid),
        )
        return {label: dict(w.axes) for label, w in zip(labels, selected)}

    def _mark_representatives(
        self, workloads: list, results: list[WorkloadResult]
    ) -> list[WorkloadResult]:
        names_by_uuid = {
            workload_uuid: name
            for name, workload_uuid in self.representative_workloads.items()
        }
        for workload, result in zip(workloads, results):
            result.representative_name = names_by_uuid.get(str(workload.uuid))
        return results

    def build_contract(self) -> str | None:
        spec = read_build_spec(self.workspace)
        return _build_contract_text(spec, self.definition) if spec else None

    def task_spec(self) -> TaskSpec:
        """Map the SOL Definition into the neutral TaskSpec. SOL's tolerance is
        per-workload, and TaskSpec's is the run's single bar — so it is reported
        only when every workload agrees on it."""
        d = self.definition.model_dump(mode="json")
        return TaskSpec(
            name=d["name"],
            description=d.get("description") or "",
            op_type=d.get("op_type"),
            axes={k: _axis_field(v) for k, v in (d.get("axes") or {}).items()},
            inputs={k: _tensor_field(v) for k, v in (d.get("inputs") or {}).items()},
            outputs={k: _tensor_field(v) for k, v in (d.get("outputs") or {}).items()},
            reference=d.get("reference") or "",
            constraints=d.get("constraints") or [],
            tolerance=self._shared_tolerance(),
        )

    def _shared_tolerance(self) -> Tolerance | None:
        tolerances = {w.tolerance.model_dump_json() for w in self.workloads}
        if len(tolerances) != 1:
            return None
        t = self.workloads[0].tolerance
        return Tolerance(
            max_atol=t.max_atol,
            max_rtol=t.max_rtol,
            required_matched_ratio=t.required_matched_ratio,
        )

    def strip_build_noise(self, text: str) -> str:
        return _strip_build_noise(text)

    def prewarm(self) -> None:
        """Compile the current kernel to a reusable on-disk artifact so a separate
        ncu child reuses it instead of recompiling under instrumentation."""
        _stage_and_compile(
            self.workspace,
            self.definition,
            self._build_solution(),
            self.workloads[:1],
        )

    def build_profilable(self, label: str) -> tuple[Callable, list]:
        """Build the current kernel and materialize one representative workload's
        complete call arguments. Returns (runnable, arguments); call
        runnable(*arguments).

        Loads the compiled artifact directly rather than running SOL's evaluator:
        that keeps CUPTI — SOL's timer — out of the profiled process. A profiler
        and CUPTI cannot coexist (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED),
        and the loser degrades silently, so measuring nothing looks like success."""
        workload = representative_item_for_label(
            self.workloads,
            label,
            self.representative_workloads,
            lambda workload: str(workload.uuid),
        )
        solution = self._build_solution()
        staging = _stage_and_compile(
            self.workspace,
            self.definition,
            solution,
            [workload],
        )
        inputs = _materialize_inputs(
            self.workspace,
            self.definition,
            workload,
            reference_dir=staging,
        )
        arguments = _profile_arguments(
            self.definition,
            workload,
            inputs,
            destination_passing_style=solution.spec.destination_passing_style,
        )
        return _load_runnable(solution, staging), arguments

    def _build_solution(self):
        files = read_src_files(self.workspace)
        if not files:
            raise RuntimeError("the working kernel has no source files")
        spec = read_build_spec(self.workspace)
        if spec is None:
            raise RuntimeError("the working kernel has no build contract")
        return _solution_with_spec(self.definition.name, files, spec)


def _is_cpp(solution) -> bool:
    """Whether SOL compiles this solution, rather than importing it."""
    return bool(set(solution.spec.languages) & set(CPP_LANGUAGES))


def _subprocess_env(workspace: Path) -> dict:
    """SOL never stages safetensors; it resolves them against this root, and the
    fixture's blob paths are relative to the task."""
    return {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "FLASHINFER_TRACE_DIR": str(workspace / TASK_DIR),
    }


# --- Profiling support (build once, load in-process, no CUPTI) ---
def _stage_and_compile(
    workspace: Path,
    definition,
    solution,
    workloads,
) -> Path:
    """Stage the problem and compile its artifact into a directory keyed by the
    kernel's content hash, so the parent's prewarm and the profiler's child share
    one build. Only the current kernel's build is kept."""
    profile_build_dir = workspace / PROFILE_BUILD_DIR
    staging = profile_build_dir / solution.name
    if (staging / "benchmark_kernel.so").exists():
        return staging

    from sol_execbench import BenchmarkConfig
    from sol_execbench.driver.problem_packager import ProblemPackager

    profile_build_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_builds(profile_build_dir, keep=solution.name)
    packager = ProblemPackager(
        definition=definition,
        workloads=list(workloads),
        solution=solution,
        config=BenchmarkConfig(),
        output_dir=staging,
        keep_output_dir=True,  # the artifact must outlive this process
    )
    if not _is_cpp(solution):
        return staging  # python/triton: the packager already wrote the sources

    cmd, _ = packager.compile()
    proc = _run(
        cmd,
        str(staging),
        _subprocess_env(workspace),
        COMPILE_TIMEOUT_S,
    )
    if proc is None:
        raise RuntimeError("compiling the working kernel timed out")
    if proc.returncode != 0:
        raise RuntimeError(_compile_diagnostic(proc))
    return staging


def _prune_stale_builds(profile_build_dir: Path, keep: str) -> None:
    if not profile_build_dir.is_dir():
        return
    for path in profile_build_dir.iterdir():
        if path.name != keep and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _load_runnable(solution, staging: Path):
    """Load the built kernel's entry point, the way SOL's own driver does."""
    import importlib.util

    entry_file, symbol = solution.spec.entry_point.rsplit("::", 1)
    if _is_cpp(solution):
        spec = importlib.util.spec_from_file_location(
            "benchmark_kernel", staging / "benchmark_kernel.so"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import importlib

        sys.path.insert(0, str(staging.resolve()))
        module = importlib.import_module(Path(entry_file).stem)
    return getattr(module, symbol)


def _materialize_inputs(
    workspace: Path,
    definition,
    workload,
    device: str = "cuda:0",
    reference_dir: Path | None = None,
) -> list:
    """Input values in definition order — call the runnable as runnable(*inputs)."""
    from sol_execbench.core.bench.io import gen_inputs, load_safetensors

    custom_inputs_fn = None
    custom_inputs_entrypoint = getattr(definition, "custom_inputs_entrypoint", None)
    if custom_inputs_entrypoint is not None:
        if reference_dir is None:
            raise RuntimeError(
                "materializing custom inputs requires a reference module directory"
            )
        custom_inputs_fn = _load_reference_entrypoint(
            definition,
            custom_inputs_entrypoint,
            reference_dir,
        )

    safe_tensors = (
        load_safetensors(definition, workload, [workspace / TASK_DIR])
        if any(v.type == "safetensors" for v in workload.inputs.values())
        else None
    )
    return gen_inputs(
        definition,
        workload,
        device=device,
        safe_tensors=safe_tensors,
        custom_inputs_fn=custom_inputs_fn,
    )


def _profile_arguments(
    definition,
    workload,
    inputs: list,
    destination_passing_style: bool,
    device: str = "cuda:0",
) -> list:
    """Complete a profiler invocation with outputs allocated outside NCU."""
    if not destination_passing_style:
        return inputs

    from sol_execbench.core.bench.io import allocate_outputs

    resolved_axes = definition.get_resolved_axes_values(workload.axes)
    outputs = allocate_outputs(definition, resolved_axes, device)
    return [*inputs, *outputs]


def _load_reference_entrypoint(
    definition,
    entrypoint: str,
    reference_dir: Path,
) -> Callable:
    """Load trusted reference code exactly as SOL's evaluation driver does.

    The source is written to a real file because references may contain
    decorators (for example ``@triton.jit``) that inspect their source file.
    Profiling bypasses SOL's evaluation driver to avoid a second CUPTI
    subscriber, so the adapter must perform this step itself.
    """
    import importlib.util

    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / "_profile_reference.py"
    reference_path.write_text(definition.reference)
    module_name = f"_profile_reference_{definition.name}"
    spec = importlib.util.spec_from_file_location(module_name, reference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load reference module from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"failed to execute reference code: {exc}") from exc

    fn = getattr(module, entrypoint, None)
    if not callable(fn):
        raise RuntimeError(
            f"reference code does not define callable {entrypoint!r}"
        )
    return fn


def _build_contract_text(spec: dict, definition) -> str:
    entry = spec.get("entry_point", "")
    symbol = entry.split("::", 1)[1] if "::" in entry else entry
    outputs = list(getattr(definition, "outputs", {}) or {})
    out_str = ", ".join(outputs) if outputs else "its outputs"
    if spec.get("destination_passing_style", True):
        convention = f"writes {out_str} into caller-provided out-parameters"
    else:
        convention = f"returns ({out_str}) by value"
    parts = [f"The working kernel must define `{symbol}`, which {convention}."]
    parts.append(
        "Timed calls use new `data_ptr()` values each iteration. The score measures "
        "GPU work launched by `run`; ordinary CPU work is excluded. Reuse only "
        "pointer-independent state, and recompute outputs every call."
    )
    deps = spec.get("dependencies") or []
    if deps:
        parts.append(f"Available build dependencies: {', '.join(deps)}.")
    options = spec.get("compile_options") or {}
    if options.get("cflags"):
        parts.append(f"C++ compiler flags: {' '.join(options['cflags'])}.")
    if options.get("cuda_cflags"):
        parts.append(f"CUDA compiler flags: {' '.join(options['cuda_cflags'])}.")
    if options.get("ld_flags"):
        parts.append(f"Linker flags: {' '.join(options['ld_flags'])}.")
    return " ".join(parts)


# --- Task fixtures ---
def _load_task(workspace: Path) -> tuple[object, list]:
    """Parse task/definition.json + task/workloads.jsonl. The fixture stores one
    Trace-shaped record per line; SOL wants the bare Workload inside it."""
    from sol_execbench import Definition, Workload

    task_dir = workspace / TASK_DIR
    definition = Definition.model_validate_json(
        (task_dir / "definition.json").read_text()
    )
    workloads = [
        Workload(**json.loads(line)["workload"])
        for line in (task_dir / "workloads.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return definition, workloads


def _solution_with_spec(definition_name: str, files: list[tuple[str, str]], spec: dict):
    from sol_execbench import BuildSpec, Solution, SourceFile

    return Solution(
        name=solution_name_from_src_files(files),
        definition=definition_name,
        author=AUTHOR,
        spec=BuildSpec.model_validate(spec),
        sources=[SourceFile(path=name, content=content or "\n") for name, content in files],
    )


def _resolve_build_spec(spec: dict, overrides: dict) -> dict:
    """Apply options through SOL's native BuildSpec schema."""
    from sol_execbench import BuildSpec

    merged = {**spec, **overrides}
    if isinstance(spec.get("compile_options"), dict) and isinstance(
        overrides.get("compile_options"),
        dict,
    ):
        merged["compile_options"] = {
            **spec["compile_options"],
            **overrides["compile_options"],
        }
    return BuildSpec.model_validate(
        merged,
        extra="forbid",
    ).model_dump(mode="json")


# --- Benchmark execution ---
def _evaluate(
    workspace: Path,
    definition,
    solution,
    workloads,
    reference_timing: bool,
    archive: bool = True,
) -> list[WorkloadResult]:
    """Drive SOL's packager: stage, compile (C++ only), run the eval driver, parse
    Traces from its stdout, and map them to neutral leaves. Raises
    BenchmarkUnavailable when the benchmark, not the kernel, is what failed."""
    from sol_execbench import BenchmarkConfig
    from sol_execbench.driver.problem_packager import ProblemPackager

    is_cpp = _is_cpp(solution)
    env = _subprocess_env(workspace)

    with tempfile.TemporaryDirectory(prefix="sol_execbench_") as staging:
        packager = ProblemPackager(
            definition=definition,
            workloads=workloads,
            solution=solution,
            config=BenchmarkConfig(benchmark_reference=reference_timing),
            output_dir=Path(staging),
            keep_output_dir=True,  # the context manager owns the directory
        )

        if is_cpp:
            cmd, _ = packager.compile()
            proc = _run(cmd, staging, env, COMPILE_TIMEOUT_S)
            if proc is None:
                return _synthetic_leaves(workloads, "COMPILE_ERROR", "compilation timed out")
            if proc.returncode != 0:
                # SOL's CLI just exits here; the contract wants one leaf per workload.
                return _synthetic_leaves(workloads, "COMPILE_ERROR", _compile_diagnostic(proc))

        proc = _run(packager.execute(), staging, env, EVAL_TIMEOUT_S)
        if proc is None:
            return _synthetic_leaves(workloads, "TIMEOUT", "evaluation timed out")
        if proc.returncode != 0 and not proc.stdout.strip():
            raise BenchmarkUnavailable(f"the benchmark failed to run: {_tail(proc.stderr)}")

        traces = packager.convert_stdout_to_traces(proc.stdout)

    if not traces:
        raise BenchmarkUnavailable(f"the benchmark produced no results: {_tail(proc.stderr)}")
    if reason := _infra_reason(traces):
        raise BenchmarkUnavailable(f"the benchmark cannot measure the kernel: {reason}")

    if archive:
        _append_traces_to_archive(workspace, traces)
    return _workload_results(traces)


def _run(cmd: list[str], cwd: str, env: dict, timeout: int):
    """Run one SOL subprocess; None on timeout."""
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None


def _infra_reason(traces) -> str | None:
    """The first trace whose failure is the benchmark's fault, not the kernel's."""
    for trace in traces:
        log = (getattr(trace.evaluation, "log", "") or "").strip()
        if log.startswith(INFRA_LOG_PREFIXES):
            return log.splitlines()[0]
    return None


def _synthetic_leaves(workloads, outcome: str, diagnostic: str) -> list[WorkloadResult]:
    """One failing leaf per workload, for failures SOL reports without traces."""
    return [
        WorkloadResult(axes=dict(w.axes), outcome=outcome, diagnostic=diagnostic or None)
        for w in workloads
    ]


def _tail(*texts: str, lines: int = 40) -> str:
    for text in texts:
        if text and text.strip():
            return "\n".join(text.strip().splitlines()[-lines:])
    return ""


def _compile_diagnostic(proc) -> str:
    """The compiler's own diagnostic. nvcc writes it to stdout, buried under
    ninja's command echoes; stderr only carries torch's cpp_extension traceback,
    so it is a fallback, not the source."""
    return denoise(proc.stdout, compile_error=True) or denoise(proc.stderr)


# --- Archive I/O ---
def _append_solution_to_archive(workspace: Path, solution) -> None:
    """Append `solution` to archive/solutions.jsonl, deduped by name."""
    path = _archive_dir(workspace) / "solutions.jsonl"
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                continue
    if solution.name in seen:
        return
    with path.open("a") as f:
        f.write(solution.model_dump_json() + "\n")


def _append_traces_to_archive(workspace: Path, traces) -> None:
    if not traces:
        return
    with (_archive_dir(workspace) / "traces.jsonl").open("a") as f:
        for t in traces:
            f.write(t.model_dump_json() + "\n")


def _archive_dir(workspace: Path) -> Path:
    archive_dir = workspace / ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir


# --- Neutral leaf mapping ---
def _workload_results(traces) -> list[WorkloadResult]:
    return [_workload_result(t) for t in traces]


def _workload_result(trace) -> WorkloadResult:
    evaluation = trace.evaluation
    passed = evaluation.status.value == "PASSED"
    perf = evaluation.performance if passed else None
    has_reference_timing = bool(
        perf
        and perf.reference_latency_ms is not None
        and perf.reference_latency_ms > 0
        and perf.speedup_factor is not None
        and perf.speedup_factor > 0
    )
    return WorkloadResult(
        axes=dict(trace.workload.axes),
        outcome=normalize_outcome(evaluation.status.value),
        latency_ms=round(perf.latency_ms, 6) if perf else None,
        reference_latency_ms=(
            round(perf.reference_latency_ms, 6)
            if has_reference_timing
            else None
        ),
        speedup_factor=(
            round(perf.speedup_factor, 4) if has_reference_timing else None
        ),
        tolerance=_tolerance(trace.workload.tolerance),
        correctness=_correctness(evaluation),
        diagnostic=None if passed else _workload_diagnostic(trace),
    )


def _workload_diagnostic(trace) -> str | None:
    """Compose a correctness summary and the driver's log into one note. SOL emits
    no message at all on a numerical failure — only the error metrics — so without
    this the agent gets bare maxima and no idea what bar they missed."""
    parts = []
    text = _correctness_text(trace.evaluation, trace.workload.tolerance)
    if text:
        parts.append(text)
    tail = denoise(trace.evaluation.log or "")
    if tail:
        parts.append(tail)
    return "\n".join(parts) or None


def _correctness_text(evaluation, tolerance) -> str | None:
    """State each error against its bound and the gate that used it. SOL's gate is
    allclose-style and ratio-based — NOT flashinfer's both-tolerances rule — so it
    must be described in SOL's own terms."""
    c = getattr(evaluation, "correctness", None)
    if c is None:
        return None
    if c.has_nan or _is(c.max_absolute_error, c.max_relative_error, test=math.isnan):
        return "output contains NaN"
    if c.has_inf or _is(c.max_absolute_error, c.max_relative_error, test=math.isinf):
        return "output contains Inf"
    return (
        f"max absolute error {_finite(c.max_absolute_error)}, "
        f"max relative error {_finite(c.max_relative_error)}; "
        f"an element fails when |error| > atol + rtol*|reference| "
        f"(atol {tolerance.max_atol}, rtol {tolerance.max_rtol}), and the workload "
        f"fails when fewer than {tolerance.required_matched_ratio:.0%} of elements match"
    )


def _correctness(evaluation) -> Correctness | None:
    """Structured correctness. Non-finite metrics become flags with the numeric
    field left null, so the leaf stays JSON-clean."""
    c = getattr(evaluation, "correctness", None)
    if c is None:
        return None
    abs_err, rel_err = c.max_absolute_error, c.max_relative_error
    return Correctness(
        max_abs_error=_finite(abs_err),
        max_rel_error=_finite(rel_err),
        has_nan=c.has_nan or _is(abs_err, rel_err, test=math.isnan),
        has_inf=c.has_inf or _is(abs_err, rel_err, test=math.isinf),
    )


def _tolerance(tolerance) -> Tolerance:
    return Tolerance(
        max_atol=tolerance.max_atol,
        max_rtol=tolerance.max_rtol,
        required_matched_ratio=tolerance.required_matched_ratio,
    )


def _is(*values, test) -> bool:
    return any(isinstance(v, float) and test(v) for v in values)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(value, 6) if isinstance(value, float) else value


def _axis_field(a: dict) -> AxisField:
    return AxisField(
        kind=a.get("type", "var"),
        value=a.get("value"),
        expression=a.get("expression"),
        description=a.get("description"),
    )


def _tensor_field(t: dict) -> TensorField:
    return TensorField(
        shape=t.get("shape"),
        dtype=t.get("dtype", "?"),
        description=t.get("description"),
    )
