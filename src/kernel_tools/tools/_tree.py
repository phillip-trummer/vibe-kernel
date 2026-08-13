"""Durable experiment graph and its readable projections."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ._workloads import REPRESENTATIVE_WORKLOAD_LABELS
from ._workspace import (
    read_benchmark_manifest,
    read_src_files,
    solution_name_from_src_files,
)


SCHEMA_VERSION = 5
MEMORY_PATH = Path(".state/memory.json")
MEMORY_VIEW_PATH = Path("experiment_memory.md")

EXPERIMENT_RE = re.compile(r"^e(\d+)$")
BRANCH_RE = re.compile(r"^b(\d+)$")
IDEA_RE = re.compile(r"^i(\d+)$")


def bootstrap_memory(
    *,
    task: str,
    kernel_description: str,
    hardware: str,
    language: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "kernel_description": kernel_description,
        "hardware": hardware,
        "language": language,
        "task_spec": {},
        "build_contract": None,
        "representative_workload_axes": {},
        "target": None,
        "active_branch": None,
        "branches": {},
        "experiments": {},
        "ideas": {},
    }


def create_memory(workspace: Path) -> dict:
    """Create initial memory from the workspace task without running a kernel."""
    from ._benchmark import get_adapter

    manifest = read_benchmark_manifest(workspace)
    build_spec = manifest["build_spec"]
    adapter = get_adapter(workspace, manifest)
    task_spec = adapter.task_spec().model_dump(mode="json")
    languages = build_spec.get("languages") or [build_spec.get("language", "")]

    memory = bootstrap_memory(
        task=task_spec["name"],
        kernel_description=task_spec.get("description", ""),
        hardware=manifest["hardware"],
        language=languages[0],
    )
    memory["task_spec"] = task_spec
    memory["build_contract"] = adapter.build_contract()
    memory["representative_workload_axes"] = adapter.representative_axes()
    return memory


def load_memory(workspace: Path) -> dict:
    memory_path = workspace / MEMORY_PATH
    if not memory_path.is_file():
        raise FileNotFoundError(
            f"{memory_path} not found; experiment memory has not been initialized."
        )
    memory = json.loads(memory_path.read_text())
    if memory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experiment memory schema "
            f"{memory.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    return memory


def load_or_initialize_memory(workspace: Path) -> dict:
    if (workspace / MEMORY_PATH).is_file():
        return load_memory(workspace)
    memory = create_memory(workspace)
    save_memory(workspace, memory)
    return memory


def save_memory(workspace: Path, memory: dict) -> None:
    memory_path = workspace / MEMORY_PATH
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(memory, indent=2) + "\n")
    (workspace / MEMORY_VIEW_PATH).write_text(render_memory(memory, workspace))


def get_active_branch(memory: dict) -> Optional[str]:
    return memory["active_branch"]


def set_active_branch(memory: dict, branch_id: str) -> None:
    memory["active_branch"] = branch_id


def get_head(memory: dict) -> Optional[str]:
    branch_id = get_active_branch(memory)
    return memory["branches"][branch_id]["head"] if branch_id else None


def has_experiment(memory: dict, experiment_id: str) -> bool:
    return experiment_id in memory["experiments"]


def list_experiment_ids(memory: dict) -> list[str]:
    return sorted(memory["experiments"], key=_experiment_order)


def has_branch(memory: dict, branch_id: str) -> bool:
    return branch_id in memory["branches"]


def list_branch_ids(memory: dict) -> list[str]:
    return sorted(memory["branches"], key=_branch_order)


def branch_history(memory: dict, branch_id: str) -> list[str]:
    """Return branch-local experiment ids newest-first."""
    branch = memory["branches"][branch_id]
    boundary = branch["forked_from"]
    experiment_id = branch["head"]
    history = []
    while experiment_id is not None and experiment_id != boundary:
        history.append(experiment_id)
        experiment_id = memory["experiments"][experiment_id]["parent_id"]
    return history


def head_state(workspace: Path, memory: dict) -> Optional[str]:
    head = get_head(memory)
    if head is None:
        return None
    solution = memory["experiments"][head]["solution"]
    working_solution = solution_name_from_src_files(read_src_files(workspace))
    return "clean" if working_solution == solution else "dirty"


def find_experiment_by_solution(memory: dict, solution_name: str) -> Optional[str]:
    for experiment_id, experiment in memory["experiments"].items():
        if experiment["solution"] == solution_name:
            return experiment_id
    return None


def next_experiment_number(memory: dict, experiments_dir: Path) -> int:
    numbers = []
    if experiments_dir.is_dir():
        for path in experiments_dir.iterdir():
            match = EXPERIMENT_RE.fullmatch(path.name) if path.is_dir() else None
            if match:
                numbers.append(int(match.group(1)))
    for experiment_id in memory["experiments"]:
        match = EXPERIMENT_RE.fullmatch(experiment_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_branch_number(memory: dict) -> int:
    numbers = []
    for branch_id in memory["branches"]:
        match = BRANCH_RE.fullmatch(branch_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_idea_number(memory: dict) -> int:
    numbers = []
    for idea_id in memory["ideas"]:
        match = IDEA_RE.fullmatch(idea_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def add_experiment(
    memory: dict,
    *,
    experiment_id: str,
    parent_id: Optional[str],
    summary: str,
    solution: str,
    evaluation: dict,
) -> None:
    memory["experiments"][experiment_id] = {
        "parent_id": parent_id,
        "summary": summary,
        "solution": solution,
        "evaluation": evaluation,
    }


def add_root_branch(memory: dict, branch_id: str, experiment_id: str) -> None:
    memory["branches"][branch_id] = {
        "forked_from": None,
        "head": experiment_id,
    }
    memory["active_branch"] = branch_id


def advance_branch(memory: dict, branch_id: str, experiment_id: str) -> None:
    memory["branches"][branch_id]["head"] = experiment_id


def fork_branch(
    memory: dict,
    branch_id: str,
    forked_from: str,
    experiment_id: str,
) -> None:
    memory["branches"][branch_id] = {
        "forked_from": forked_from,
        "head": experiment_id,
    }
    memory["active_branch"] = branch_id


def list_idea_ids(memory: dict) -> list[str]:
    return sorted(memory["ideas"], key=_idea_order)


def has_idea(memory: dict, idea_id: str) -> bool:
    return idea_id in memory["ideas"]


def add_idea(memory: dict, branch_id: str, text: str) -> str:
    idea_id = f"i{next_idea_number(memory)}"
    memory["ideas"][idea_id] = {"branch_id": branch_id, "text": text}
    return idea_id


def remove_idea(memory: dict, idea_id: str) -> None:
    del memory["ideas"][idea_id]


def ideas_for_branch(memory: dict, branch_id: str) -> list[str]:
    return [
        idea_id
        for idea_id in list_idea_ids(memory)
        if memory["ideas"][idea_id]["branch_id"] == branch_id
    ]


def better_evaluation(new: dict, current: Optional[dict]) -> bool:
    return _scorable(new) and _geomean_better(new, current)


def current_best(memory: dict) -> Optional[str]:
    best = None
    for experiment_id in list_experiment_ids(memory):
        if better_evaluation(
            memory["experiments"][experiment_id]["evaluation"],
            memory["experiments"][best]["evaluation"] if best else None,
        ):
            best = experiment_id
    return best


def best_by_representative_workload(memory: dict) -> dict[str, str]:
    bests = {}
    for experiment_id in list_experiment_ids(memory):
        evaluation = memory["experiments"][experiment_id]["evaluation"]
        for label, result in evaluation.get("representative_workloads", {}).items():
            if result["outcome"] != "PASSED":
                continue
            current_id = bests.get(label)
            current_result = (
                memory["experiments"][current_id]["evaluation"]
                .get("representative_workloads", {})
                .get(label)
                if current_id
                else None
            )
            if _result_better(result, current_result):
                bests[label] = experiment_id
    return bests


def _scorable(evaluation: dict) -> bool:
    return (
        evaluation.get("geomean_speedup_factor") is not None
        or evaluation.get("geomean_latency_ms") is not None
    )


def _geomean_better(new: dict, current: Optional[dict]) -> bool:
    return _metric_better(
        new.get("geomean_speedup_factor"),
        new.get("geomean_latency_ms"),
        (current or {}).get("geomean_speedup_factor"),
        (current or {}).get("geomean_latency_ms"),
    )


def _result_better(new: dict, current: Optional[dict]) -> bool:
    return _metric_better(
        new.get("speedup_factor"),
        new.get("latency_ms"),
        (current or {}).get("speedup_factor"),
        (current or {}).get("latency_ms"),
    )


def _metric_better(
    new_speedup: Optional[float],
    new_latency: Optional[float],
    current_speedup: Optional[float],
    current_latency: Optional[float],
) -> bool:
    if new_speedup is not None:
        return current_speedup is None or new_speedup > current_speedup
    if new_latency is None:
        return False
    return current_latency is None or new_latency < current_latency


def render_memory(memory: dict, workspace: Optional[Path] = None) -> str:
    lines = ["# Experiment Memory", ""]
    lines.extend(render_task_spec(memory))
    lines.extend(_render_results(memory))
    lines.extend(_render_current_branch(memory, workspace))
    lines.extend(_render_other_branches(memory))
    return "\n".join(lines).rstrip() + "\n"


def render_catalog_memory(workspace: Path, memory: dict) -> str:
    return render_memory(memory, workspace)


def render_branch_memory(memory: dict, branch_id: str) -> str:
    lines = ["# Experiment Memory", ""]
    lines.extend(_render_expanded_branch(memory, branch_id))
    return "\n".join(lines).rstrip() + "\n"


def render_branch_projection(memory: dict, branch_id: str) -> str:
    return "\n".join(_render_expanded_branch(memory, branch_id)).rstrip() + "\n"


def render_current_state_memory(workspace: Path, memory: dict) -> str:
    return "\n".join(_render_current_branch(memory, workspace)).rstrip() + "\n"


def _render_results(memory: dict) -> list[str]:
    lines = ["## Results", ""]
    target_label, target_evaluation = _target(memory)
    if target_label is not None:
        lines.append(
            f"- **Target:** `{target_label}` — "
            f"{_evaluation_summary(target_evaluation)}"
        )
        lines.extend(_representative_workload_lines(target_evaluation))

    best = current_best(memory)
    if best:
        evaluation = memory["experiments"][best]["evaluation"]
        lines.append(
            f"- **Global best:** `{best}` — "
            f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
        )
    else:
        lines.append("- **Global best:** _(unset)_")

    bests = best_by_representative_workload(memory)
    if bests:
        rendered = ", ".join(
            f"{label}=`{bests[label]}`"
            for label in REPRESENTATIVE_WORKLOAD_LABELS
            if label in bests
        )
        lines.append(f"- **Best by representative workload:** {rendered}")
    lines.append("")
    return lines


def _render_current_branch(
    memory: dict,
    workspace: Optional[Path],
) -> list[str]:
    lines = ["## Current branch", ""]
    branch_id = get_active_branch(memory)
    if branch_id is None:
        lines.extend(
            [
                "- **Active branch:** _(unset — log the root experiment)_",
                "- **Head:** _(unset)_",
                "",
                "### Local history",
                "",
                "_(none)_",
                "",
                "### Ideas",
                "",
                "_(none)_",
                "",
            ]
        )
        return lines

    branch = memory["branches"][branch_id]
    lines.append(f"- **Active branch:** `{branch_id}`")
    lines.append(f"- **Head:** `{branch['head']}`")
    if workspace is not None:
        lines.append(f"- **Working source:** {head_state(workspace, memory)}")
    lines.append("")
    lines.extend(_render_history(memory, branch_id))
    lines.extend(render_ideas(memory, branch_id))
    return lines


def _render_expanded_branch(memory: dict, branch_id: str) -> list[str]:
    branch = memory["branches"][branch_id]
    marker = " _(active)_" if branch_id == get_active_branch(memory) else ""
    lines = [f"## Branch `{branch_id}`{marker}", ""]
    lines.append(
        f"- **Forked from:** `{branch['forked_from']}`"
        if branch["forked_from"] is not None
        else "- **Forked from:** _(root)_"
    )
    lines.extend([f"- **Head:** `{branch['head']}`", ""])
    lines.extend(_render_history(memory, branch_id))
    lines.extend(render_ideas(memory, branch_id))
    return lines


def _render_history(memory: dict, branch_id: str) -> list[str]:
    lines = ["### Local history", ""]
    target_label, target_evaluation = _target(memory)
    head = memory["branches"][branch_id]["head"]
    for experiment_id in branch_history(memory, branch_id):
        experiment = memory["experiments"][experiment_id]
        marker = " _(head)_" if experiment_id == head else ""
        lines.append(f"#### `{experiment_id}`{marker}")
        lines.append(f"- **Summary:** {experiment['summary']}")
        lines.append(
            f"- **Evaluation:** "
            f"{_evaluation_summary(experiment['evaluation'], target_label, target_evaluation)}"
        )
        lines.extend(
            _representative_workload_lines(
                experiment["evaluation"],
                target_label,
                target_evaluation,
            )
        )
        lines.append("")
    return lines


def render_ideas(memory: dict, branch_id: str) -> list[str]:
    lines = ["### Ideas", ""]
    idea_ids = ideas_for_branch(memory, branch_id)
    if idea_ids:
        lines.extend(
            f"- `{idea_id}` — {memory['ideas'][idea_id]['text']}"
            for idea_id in idea_ids
        )
    else:
        lines.append("_(none)_")
    lines.append("")
    return lines


def _render_other_branches(memory: dict) -> list[str]:
    lines = ["## Other branches", ""]
    active = get_active_branch(memory)
    branch_ids = [
        branch_id
        for branch_id in list_branch_ids(memory)
        if branch_id != active
    ]
    if not branch_ids:
        lines.extend(["_(none)_", ""])
        return lines

    target_label, target_evaluation = _target(memory)
    for branch_id in branch_ids:
        branch = memory["branches"][branch_id]
        head = branch["head"]
        experiment = memory["experiments"][head]
        lines.append(f"### `{branch_id}`")
        lines.append(f"- **Head:** `{head}` — {experiment['summary']}")
        lines.append(
            f"- **Evaluation:** "
            f"{_evaluation_summary(experiment['evaluation'], target_label, target_evaluation)}"
        )
        idea_ids = ideas_for_branch(memory, branch_id)
        if idea_ids:
            lines.append("- **Ideas:**")
            lines.extend(
                f"  - `{idea_id}` — {memory['ideas'][idea_id]['text']}"
                for idea_id in idea_ids
            )
        lines.append("")
    return lines


def render_task_spec(memory: dict) -> list[str]:
    """Render the complete static task context shared by memory and agents."""
    spec = memory.get("task_spec") or {}
    lines = [
        "## Task specification",
        "",
        f"- **Task:** {memory['task']}",
    ]
    if memory["kernel_description"]:
        lines.append(f"- **Kernel:** {memory['kernel_description']}")
    lines.extend(
        [
            f"- **Hardware:** {memory['hardware']}",
            f"- **Language:** {memory['language']}",
        ]
    )
    axes_by_label = memory.get("representative_workload_axes") or {}
    labels = [
        label
        for label in REPRESENTATIVE_WORKLOAD_LABELS
        if axes_by_label.get(label)
    ]
    if labels:
        lines.append("- **Representative workloads:**")
        for label in labels:
            axes = ", ".join(
                f"{key}={value}" for key, value in axes_by_label[label].items()
            )
            lines.append(f"  - {label}: {axes}")
    contract = memory.get("build_contract")
    if contract:
        lines.append(f"- **Build contract:** {contract}")
    if spec.get("op_type"):
        lines.append(f"- **Operation:** {spec['op_type']}")
    tolerance = spec.get("tolerance")
    if tolerance:
        description = _tolerance_desc(tolerance)
        if description:
            lines.append(f"- **Correctness tolerance:** {description}")
    axes = spec.get("axes") or {}
    if axes:
        lines.append("- **Axes:**")
        for name, axis in axes.items():
            lines.append(f"  - `{name}`: {_axis_desc(axis)}")
    for title, key in (("Inputs", "inputs"), ("Outputs", "outputs")):
        fields = spec.get(key) or {}
        if fields:
            lines.append(f"- **{title}:**")
            for name, field in fields.items():
                lines.append(f"  - `{name}`: {_tensor_desc(field)}")
    constraints = spec.get("constraints") or []
    if constraints:
        lines.append("- **Constraints:**")
        lines.extend(f"  - {constraint}" for constraint in constraints)
    reference = spec.get("reference") or ""
    if reference:
        lines.extend(
            [
                "- **Reference implementation:**",
                "",
                "```python",
                *reference.splitlines(),
                "```",
            ]
        )
    lines.append("")
    return lines


def _axis_desc(axis: dict) -> str:
    kind = axis.get("kind", "var")
    if kind == "const":
        head = f"const = {axis.get('value')}"
    elif kind == "expr":
        expression = axis.get("expression")
        head = f"expr = {expression}" if expression else "expr"
    else:
        head = kind
    description = axis.get("description")
    return f"{head} — {description}" if description else head


def _tolerance_desc(tolerance: dict) -> str:
    parts = []
    if tolerance.get("max_rtol") is not None:
        parts.append(f"rtol {tolerance['max_rtol']}")
    if tolerance.get("max_atol") is not None:
        parts.append(f"atol {tolerance['max_atol']}")
    ratio = tolerance.get("required_matched_ratio")
    if ratio is not None:
        parts.append(f"≥{ratio:.0%} of elements within tolerance")
    return ", ".join(parts)


def _tensor_desc(field: dict) -> str:
    shape = field.get("shape")
    shape_text = f"[{', '.join(map(str, shape))}]" if shape else "scalar"
    base = f"{shape_text} {field.get('dtype', '?')}"
    description = field.get("description")
    return f"{base} — {description}" if description else base


def render_tool_result(acknowledgement: str, projection: str) -> str:
    return f"{acknowledgement.rstrip()}\n\n{projection.strip()}\n"


def _experiment_order(experiment_id: str) -> tuple[int, str]:
    match = EXPERIMENT_RE.fullmatch(experiment_id)
    return (int(match.group(1)), experiment_id) if match else (-1, experiment_id)


def _branch_order(branch_id: str) -> tuple[int, str]:
    match = BRANCH_RE.fullmatch(branch_id)
    return (int(match.group(1)), branch_id) if match else (-1, branch_id)


def _idea_order(idea_id: str) -> tuple[int, str]:
    match = IDEA_RE.fullmatch(idea_id)
    return (int(match.group(1)), idea_id) if match else (-1, idea_id)


def _target(memory: dict) -> tuple[Optional[str], Optional[dict]]:
    target = memory.get("target")
    if not target:
        return None, None
    return target["label"], target["evaluation"]


def _evaluation_summary(
    evaluation: dict,
    target_label: Optional[str] = None,
    target_evaluation: Optional[dict] = None,
) -> str:
    parts = [evaluation["status"]]
    count = evaluation.get("workload_count")
    over = f" over {count} workload{'s' if count != 1 else ''}" if count else ""
    latency = evaluation.get("geomean_latency_ms")
    if latency is not None:
        parts.append(f"geomean {_fmt_ms(latency)}{over}")
    geomean = evaluation.get("geomean_speedup_factor")
    if geomean is not None:
        parts.append(f"geomean {geomean:.2f}× vs reference{over}")
    ratio = _target_geomean_ratio(evaluation, target_evaluation)
    if target_label is not None and ratio is not None:
        parts.append(f"{ratio:.2f}× vs {target_label}")
    return "; ".join(parts)


def _fmt_ms(milliseconds: float) -> str:
    return f"{milliseconds:.4g} ms"


def _representative_workload_lines(
    evaluation: dict,
    target_label: Optional[str] = None,
    target_evaluation: Optional[dict] = None,
) -> list[str]:
    lines = []
    representatives = evaluation.get("representative_workloads", {})
    target_representatives = (
        target_evaluation.get("representative_workloads", {})
        if target_evaluation
        else {}
    )
    for label in REPRESENTATIVE_WORKLOAD_LABELS:
        if label not in representatives:
            continue
        result = representatives[label]
        outcome = result["outcome"]
        line = f"  - representative {label}: {outcome}"
        if outcome == "PASSED":
            metrics = []
            latency = result.get("latency_ms")
            if latency is not None:
                metrics.append(_fmt_ms(latency))
            speedup = result.get("speedup_factor")
            if speedup is not None:
                metrics.append(f"{speedup:.2f}× vs reference")
            ratio = _target_representative_workload_ratio(
                result,
                target_representatives.get(label),
            )
            if target_label is not None and ratio is not None:
                metrics.append(f"{ratio:.2f}× vs {target_label}")
            if metrics:
                line += f"; {'; '.join(metrics)}"
        else:
            tolerance = result.get("tolerance")
            rendered = _format_tolerance(tolerance) if tolerance else ""
            if rendered:
                line += f"; tolerance {rendered}"
        lines.append(line)
    return lines


def _format_tolerance(tolerance: dict) -> str:
    parts = []
    for key, label in (
        ("max_atol", "atol"),
        ("max_rtol", "rtol"),
        ("required_matched_ratio", "matched"),
    ):
        value = tolerance.get(key)
        if value is not None:
            parts.append(f"{label}={value:g}")
    return ", ".join(parts)


def _target_geomean_ratio(
    evaluation: dict,
    target_evaluation: Optional[dict],
) -> Optional[float]:
    if target_evaluation is None:
        return None
    return _vs_target(
        evaluation.get("geomean_speedup_factor"),
        evaluation.get("geomean_latency_ms"),
        target_evaluation.get("geomean_speedup_factor"),
        target_evaluation.get("geomean_latency_ms"),
    )


def _target_representative_workload_ratio(
    result: dict,
    target_result: Optional[dict],
) -> Optional[float]:
    if target_result is None:
        return None
    return _vs_target(
        result.get("speedup_factor"),
        result.get("latency_ms"),
        target_result.get("speedup_factor"),
        target_result.get("latency_ms"),
    )


def _vs_target(
    speedup: Optional[float],
    latency: Optional[float],
    target_speedup: Optional[float],
    target_latency: Optional[float],
) -> Optional[float]:
    if speedup is not None and target_speedup is not None:
        return _ratio(speedup, target_speedup)
    return _ratio(target_latency, latency)


def _ratio(
    numerator: Optional[float],
    denominator: Optional[float],
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator
