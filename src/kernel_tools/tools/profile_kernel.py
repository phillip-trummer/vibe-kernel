"""Run the kernel in src/ under Nsight Compute (ncu).

The registered function shells out through the private ``_profile_runner``
module. That child asks the adapter to build the kernel and materialize one
representative workload's inputs, warms up, then runs measured iterations
inside cudaProfilerStart/Stop.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

from kernel_tools.registry import registry

from ._benchmark import get_adapter
from ._workloads import REPRESENTATIVE_WORKLOAD_LABELS


_WARMUP_ITERS = 5
_PROFILE_ITERS = 1
_MAX_OUTPUT_BYTES = 50_000
_NCU_DEFAULTS = {
    "--cache-control": "all",
    "--clock-control": "boost",
    "--replay-mode": "kernel",
}


SCHEMA = {
    "name": "profile_kernel",
    "description": (
        "Profile the current working kernel with Nsight Compute, or introspect ncu. "
        "Pass representative_workload to profile one concrete representative workload "
        "(small, medium, large, or xlarge) and return ncu's report with a short "
        "profiling-context header. Use benchmark_kernel to rank variants: ncu measures "
        "individual kernels under profiler cache, clock, and replay policies, so its "
        "duration is not directly comparable to benchmark_kernel's full-call latency. "
        "Omit representative_workload to run ncu with no target "
        "— for informational flags like `--help`, `--list-sets`, `--list-sections`, "
        "`--query-metrics`. ncu_args are forwarded as-is to ncu. "
        "For application-realistic timing of a kernel that consumes cache state "
        "produced by earlier kernels, consider `--cache-control none`; ncu otherwise "
        "flushes GPU caches before replay. This is best suited to single-pass metric "
        "collections, since preserving caches can make replayed metrics less reproducible. "
        f"Output is capped at ~{_MAX_OUTPUT_BYTES // 1000} KB; if truncated, narrow "
        "the run with ncu flags."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "representative_workload": {
                "type": "string",
                "enum": list(REPRESENTATIVE_WORKLOAD_LABELS),
                "description": (
                    "Which representative workload to profile. "
                ),
            },
            "ncu_args": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": (
                    "Flags forwarded verbatim to ncu. For application-realistic "
                    "warm-cache timing, pass `[\"--cache-control\", \"none\"]`, preferably "
                    "with metrics that require only one replay pass."
                ),
            },
        },
    },
}


@registry.register(SCHEMA)
def profile_kernel(
    workspace: Path,
    representative_workload: str | None = None,
    ncu_args: list[str] | None = None,
) -> str:
    # The profiler child runs with cwd=workspace. Pass it an absolute path so a
    # direct caller may safely supply a relative workspace path.
    workspace = workspace.resolve()
    if shutil.which("ncu") is None:
        return "Error: ncu (Nsight Compute) not found on PATH"

    adapter = None
    if representative_workload is None:
        # Introspection mode: run ncu with no target app (e.g. --help, --list-sets).
        cmd = ["ncu", *(ncu_args or [])]
    else:
        # Validate input, then re-enter this file under ncu.
        if representative_workload not in REPRESENTATIVE_WORKLOAD_LABELS:
            valid = ", ".join(REPRESENTATIVE_WORKLOAD_LABELS)
            return (
                f"Error: invalid representative_workload "
                f"{representative_workload!r}; expected one of: {valid}"
            )
        # Pre-build so the ncu child reuses the on-disk artifact instead of
        # recompiling under instrumentation. Best-effort: on failure the child
        # recompiles and reports it. The adapter is reused below to strip build
        # chatter from the report.
        adapter = get_adapter(workspace)
        try:
            adapter.prewarm()
        except Exception:
            pass
        cmd = [
            "ncu",
            "--target-processes", "all",
            "--profile-from-start", "off",
            *(ncu_args or []),
            sys.executable,
            "-m",
            "kernel_tools.tools._profile_runner",
            str(workspace),
            representative_workload,
        ]

    result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)

    # On a profiled success, strip the backend's build chatter so the report
    # dominates. Otherwise keep everything verbatim — a failed run's build log is
    # the diagnostic, and introspection output has no chatter to strip.
    if result.returncode == 0 and adapter is not None:
        out = adapter.strip_build_noise(result.stdout)
        err = adapter.strip_build_noise(result.stderr)
        if err.strip():
            out += "\n--- stderr ---\n" + err
    else:
        out = result.stdout
        if result.stderr:
            out += "\n--- stderr ---\n" + result.stderr
    if representative_workload is None:
        return _tail_cap(out, _MAX_OUTPUT_BYTES)
    header = _profile_context_header(ncu_args or [], out)
    body_limit = max(1, _MAX_OUTPUT_BYTES - len(header.encode("utf-8")))
    return header + _tail_cap(out, body_limit)


def _profile_context_header(ncu_args: list[str], report: str) -> str:
    settings = {
        option.removeprefix("--"): _ncu_option_value(ncu_args, option, default)
        for option, default in _NCU_DEFAULTS.items()
    }
    passes = _observed_replay_passes(report)
    pass_text = ", ".join(f"{name}:{count}" for name, count in passes)
    if not pass_text:
        pass_text = "not reported by ncu"
    return (
        "[profiling context]\n"
        f"cache-control={settings['cache-control']}; "
        f"clock-control={settings['clock-control']}; "
        f"replay-mode={settings['replay-mode']}; "
        f"profile-iterations={_PROFILE_ITERS}; observed-replay-passes={pass_text}\n"
        "Use benchmark_kernel to rank end-to-end variants. NCU kernel duration is "
        "measured under the settings above and is not directly comparable to "
        "benchmark_kernel full-call latency.\n"
        "[/profiling context]\n\n"
    )


def _ncu_option_value(args: list[str], option: str, default: str) -> str:
    value = None
    prefix = option + "="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
        elif arg == option and index + 1 < len(args):
            value = args[index + 1]
    return value if value is not None else f"{default} (ncu default)"


def _observed_replay_passes(report: str) -> list[tuple[str, int]]:
    observed: list[tuple[str, int]] = []
    for line in report.splitlines():
        match = re.search(
            r'Profiling "([^"]+)".*-\s+(\d+)\s+pass(?:es)?\s*$',
            line,
        )
        if match:
            observed.append((match.group(1), int(match.group(2))))
    return observed


def _tail_cap(text: str, limit: int) -> str:
    data = text.encode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    dropped = len(data) - limit
    kept = data[-limit:].decode("utf-8", errors="replace")
    return (
        f"[truncated: dropped {dropped} bytes from start of output; "
        f"narrow ncu_args (--set basic, --section, -k, --launch-count) "
        f"to fit under {limit // 1000} KB]\n" + kept
    )


def _run_under_ncu(workspace: Path, representative_workload: str) -> None:
    import torch

    # Build the kernel + materialize the chosen representative workload's complete
    # call arguments (inputs followed by outputs for destination-passing kernels).
    # Same kernel identity as benchmark_kernel, so the adapter's build cache is shared.
    runnable, arguments = get_adapter(workspace).build_profilable(
        representative_workload
    )

    with torch.no_grad():
        # Warm up outside the profiled region.
        for _ in range(_WARMUP_ITERS):
            runnable(*arguments)
        torch.cuda.synchronize()

        # Profile.
        torch.cuda.profiler.start()
        for _ in range(_PROFILE_ITERS):
            runnable(*arguments)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
