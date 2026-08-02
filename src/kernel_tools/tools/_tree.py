"""Durable state and rendering for the experiment memory."""
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


SCHEMA_VERSION = 4
MEMORY_PATH = Path(".state/memory.json")
MEMORY_VIEW_PATH = Path("experiment_memory.md")

EXPERIMENT_RE = re.compile(r"^e(\d+)_")
BRANCH_RE = re.compile(r"^b(\d+)_")
MEMORY_ENTRY_RE = re.compile(r"^[a-z]+(\d+)$")

_MEMORY_KEY = {
    "hypothesis": "hypotheses",
    "fact": "facts",
    "hazard": "hazards",
}
_MEMORY_PREFIX = {
    "hypothesis": "h",
    "fact": "f",
    "hazard": "r",
}
_TOP_LEVEL_KEY = {
    scope: key for scope, key in _MEMORY_KEY.items() if scope != "hypothesis"
}

TOP_LEVEL_SCOPES = tuple(_TOP_LEVEL_KEY)
FINDING_SCOPES = ("experiment_finding",)


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
        "head": None,
        "active_branch": None,
        "representative_workload_axes": {},
        "target": None,
        "hypotheses": {},
        "branches": {},
        "experiments": {},
        "facts": {},
        "hazards": {},
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
    # Load current state
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
    memory_path = workspace / MEMORY_PATH
    if memory_path.is_file():
        return load_memory(workspace)

    memory = create_memory(workspace)
    save_memory(workspace, memory)
    return memory


def save_memory(workspace: Path, memory: dict) -> None:
    # Persist canonical state
    memory_path = workspace / MEMORY_PATH
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(memory, indent=2) + "\n")

    # Render readable memory
    (workspace / MEMORY_VIEW_PATH).write_text(render_memory(memory))


def get_head(memory: dict) -> Optional[str]:
    return memory["head"]


def get_experiment(memory: dict, experiment_id: str) -> dict:
    return memory["experiments"][experiment_id]


def has_experiment(memory: dict, experiment_id: str) -> bool:
    return experiment_id in memory["experiments"]


def list_experiment_ids(memory: dict) -> list[str]:
    return sorted(memory["experiments"], key=_experiment_order)


def has_branch(memory: dict, branch_id: str) -> bool:
    return branch_id in memory["branches"]


def list_branch_ids(memory: dict) -> list[str]:
    return sorted(memory["branches"], key=_branch_order)


def list_hypothesis_ids(memory: dict) -> list[str]:
    return list_memory_entry_ids(memory, "hypothesis")


def has_hypothesis(memory: dict, hypothesis_id: str) -> bool:
    return hypothesis_id in memory["hypotheses"]


def get_hypothesis(memory: dict, hypothesis_id: str) -> dict:
    return memory["hypotheses"][hypothesis_id]


def list_memory_entry_ids(memory: dict, scope: str) -> list[str]:
    return sorted(memory[_MEMORY_KEY[scope]], key=_memory_entry_order)


def has_memory_entry(memory: dict, scope: str, entry_id: str) -> bool:
    return entry_id in memory[_MEMORY_KEY[scope]]


def find_experiment_by_solution(memory: dict, solution_name: str) -> Optional[str]:
    for experiment_id, experiment in memory["experiments"].items():
        if experiment["solution"] == solution_name:
            return experiment_id
    return None


def find_branch_for_experiment(memory: dict, experiment_id: str) -> Optional[str]:
    for branch_id, branch in memory["branches"].items():
        if experiment_id in branch["experiments"]:
            return branch_id
    return None


def parent_branch_id(memory: dict, branch_id: str) -> Optional[str]:
    base = memory["branches"][branch_id]["base"]
    return find_branch_for_experiment(memory, base) if base else None


def get_active_branch(memory: dict) -> Optional[str]:
    return memory["active_branch"]


def set_active_branch(memory: dict, branch_id: str) -> None:
    memory["active_branch"] = branch_id


def branch_accepts_head(
    memory: dict,
    branch_id: str,
    experiment_id: Optional[str],
) -> bool:
    branch = memory["branches"][branch_id]
    return experiment_id == branch["base"] or experiment_id in branch["experiments"]


def head_state(workspace: Path, memory: dict) -> Optional[str]:
    head = memory["head"]
    experiment = memory["experiments"].get(head) if head else None
    solution = experiment.get("solution") if experiment else None
    files = read_src_files(workspace) if solution else []
    if not solution or not files:
        return None
    return (
        "clean" if solution_name_from_src_files(files) == solution else "dirty"
    )


def next_experiment_number(memory: dict, experiments_dir: Path) -> int:
    numbers: list[int] = []
    if experiments_dir.is_dir():
        for path in experiments_dir.iterdir():
            match = EXPERIMENT_RE.match(path.name) if path.is_dir() else None
            if match:
                numbers.append(int(match.group(1)))
    for experiment_id in memory["experiments"]:
        match = EXPERIMENT_RE.match(experiment_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_branch_number(memory: dict) -> int:
    numbers = []
    for branch_id in memory["branches"]:
        match = BRANCH_RE.match(branch_id)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_memory_entry_number(memory: dict, scope: str) -> int:
    numbers = []
    prefix = _MEMORY_PREFIX[scope]
    for entry_id in memory[_MEMORY_KEY[scope]]:
        match = MEMORY_ENTRY_RE.fullmatch(entry_id)
        if match and entry_id.startswith(prefix):
            numbers.append(int(match.group(1)))
    return max(numbers) + 1 if numbers else 0


def next_memory_entry_id(memory: dict, scope: str) -> str:
    return f"{_MEMORY_PREFIX[scope]}{next_memory_entry_number(memory, scope)}"


def add_branch(
    memory: dict,
    *,
    branch_id: str,
    base: Optional[str],
    structure: str,
) -> None:
    memory["branches"][branch_id] = {
        "base": base,
        "structure": structure,
        "experiments": [],
    }


def add_experiment(
    memory: dict,
    *,
    experiment_id: str,
    branch_id: str,
    solution: str,
    variant: str,
    finding: Optional[str],
    evaluation: dict,
) -> None:
    memory["experiments"][experiment_id] = {
        "solution": solution,
        "variant": variant,
        "finding": finding,
        "evaluation": evaluation,
    }
    memory["branches"][branch_id]["experiments"].append(experiment_id)


def set_head(memory: dict, experiment_id: str) -> None:
    memory["head"] = experiment_id


def add_hypothesis(memory: dict, base: str, text: str) -> str:
    hypothesis_id = next_memory_entry_id(memory, "hypothesis")
    memory["hypotheses"][hypothesis_id] = {"base": base, "text": text}
    return hypothesis_id


def replace_hypothesis(
    memory: dict,
    hypothesis_id: str,
    base: str,
    text: str,
) -> None:
    memory["hypotheses"][hypothesis_id] = {"base": base, "text": text}


def consume_hypothesis(memory: dict, hypothesis_id: str) -> None:
    del memory["hypotheses"][hypothesis_id]


def better_evaluation(new: dict, current: Optional[dict]) -> bool:
    """Compare scored aggregate evaluations."""
    return _scorable(new) and _geomean_better(new, current)


def branch_representative(memory: dict, branch_id: str) -> Optional[str]:
    representative = None
    for experiment_id in memory["branches"][branch_id]["experiments"]:
        if representative is None or _preferred_representative(
            memory["experiments"][experiment_id]["evaluation"],
            memory["experiments"][representative]["evaluation"],
        ):
            representative = experiment_id
    return representative


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
    bests: dict[str, str] = {}
    for experiment_id in list_experiment_ids(memory):
        evaluation = memory["experiments"][experiment_id]["evaluation"]
        for label, result in evaluation["representative_workloads"].items():
            if result["outcome"] != "PASSED":
                continue
            current_id = bests.get(label)
            current_result = (
                memory["experiments"][current_id]["evaluation"][
                    "representative_workloads"
                ].get(label)
                if current_id
                else None
            )
            if _result_better(result, current_result):
                bests[label] = experiment_id
    return bests


def _preferred_representative(new: dict, current: dict) -> bool:
    # A scored experiment always wins; between unscored ones the latest stands.
    if not _scorable(new):
        return not _scorable(current)
    return better_evaluation(new, current) or not _scorable(current)


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


def add_annotation(
    memory: dict,
    scope: str,
    text: str,
) -> str:
    entry_id = next_memory_entry_id(memory, scope)
    memory[_TOP_LEVEL_KEY[scope]][entry_id] = text
    return entry_id


def replace_annotation(
    memory: dict,
    scope: str,
    entry_id: str,
    text: str,
) -> None:
    memory[_TOP_LEVEL_KEY[scope]][entry_id] = text


def remove_annotation(memory: dict, scope: str, entry_id: str) -> None:
    del memory[_TOP_LEVEL_KEY[scope]][entry_id]


def get_finding(memory: dict, scope: str, target_id: str) -> Optional[str]:
    return memory["experiments"][target_id].get("finding")


def set_finding(
    memory: dict,
    scope: str,
    target_id: str,
    finding: Optional[str],
) -> None:
    memory["experiments"][target_id]["finding"] = finding


def render_memory(memory: dict) -> str:
    lines = ["# Experiment Memory", ""]
    lines.extend(_render_header(memory))
    lines.extend(render_task_spec(memory))
    lines.extend(
        _render_current_state(
            memory,
            expand_head=False,
            check_dirty=False,
        )
    )
    lines.extend(_render_branch_collection(memory, list_branch_ids(memory), full=True))
    return "\n".join(lines).rstrip() + "\n"


def render_catalog_memory(workspace: Path, memory: dict) -> str:
    lines = ["# Experiment Memory", ""]
    lines.extend(_render_header(memory))
    lines.extend(render_task_spec(memory))
    lines.extend(_render_current_state(memory, workspace=workspace))
    branch_ids = list_branch_ids(memory)
    lines.extend(_render_branch_collection(memory, branch_ids, full=False))
    lines.append(
        f"_({len(branch_ids)} structure"
        f"{'s' if len(branch_ids) != 1 else ''} shown; pass branch_id to "
        "read_memory to inspect all variants of one branch.)_"
    )
    return "\n".join(lines).rstrip() + "\n"


def render_branch_memory(memory: dict, branch_id: str) -> str:
    projection = render_branch_projection(memory, branch_id)
    return f"# Experiment Memory\n\n{projection.strip()}\n"


def render_branch_projection(memory: dict, branch_id: str) -> str:
    lines = _render_branch_collection(
        memory,
        [branch_id],
        full=True,
        title="Branch variants",
    )
    return "\n".join(lines).rstrip() + "\n"


def render_experiment_memory(memory: dict, experiment_id: str) -> str:
    target_label, target_evaluation = _target(memory)
    marker = "head" if experiment_id == memory["head"] else ""
    lines = ["## Experiment", ""]
    lines.extend(
        _render_experiment(
            experiment_id,
            memory["experiments"][experiment_id],
            target_label,
            target_evaluation,
            heading_level=3,
            marker=marker,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def render_current_state_memory(workspace: Path, memory: dict) -> str:
    return (
        "\n".join(_render_current_state(memory, workspace=workspace)).rstrip() + "\n"
    )


def _render_branch_collection(
    memory: dict,
    branch_ids: list[str],
    *,
    full: bool,
    title: str = "Structural catalog",
) -> list[str]:
    lines = [f"## {title}", ""]
    if not branch_ids:
        lines.extend(["_(none)_", ""])
        return lines
    for branch_id in branch_ids:
        lines.extend(_render_branch(memory, branch_id, full=full))
        lines.append("")
    return lines


def _render_header(memory: dict) -> list[str]:
    lines = [f"- **Task:** {memory['task']}"]
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
    lines.append("")

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
            f"- **Current best:** `{best}` — "
            f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
        )
    else:
        lines.append("- **Current best:** _(unset)_")

    bests = best_by_representative_workload(memory)
    if bests:
        rendered = ", ".join(
            f"{label}=`{bests[label]}`"
            for label in REPRESENTATIVE_WORKLOAD_LABELS
            if label in bests
        )
        lines.append(f"- **Best by representative workload:** {rendered}")

    lines.append("")

    for scope in ("hypothesis", "fact", "hazard"):
        lines.extend(_render_annotation_section(memory, scope))
    return lines


def _render_annotation_section(memory: dict, scope: str) -> list[str]:
    title = {
        "hypothesis": "Open hypotheses",
        "fact": "Facts",
        "hazard": "Hazards",
    }[scope]
    lines = [f"## {title}", ""]
    entry_ids = list_memory_entry_ids(memory, scope)
    if not entry_ids:
        lines.extend(["_(none)_", ""])
        return lines
    if scope == "hypothesis":
        for entry_id in entry_ids:
            hypothesis = memory["hypotheses"][entry_id]
            lines.append(
                f"- `{entry_id}` from `{hypothesis['base']}` — "
                f"{hypothesis['text']}"
            )
    else:
        items = memory[_TOP_LEVEL_KEY[scope]]
        lines.extend(f"- `{entry_id}` — {items[entry_id]}" for entry_id in entry_ids)
    lines.append("")
    return lines


def _render_current_state(
    memory: dict,
    *,
    workspace: Path | None = None,
    expand_head: bool = True,
    check_dirty: bool = True,
) -> list[str]:
    lines = ["## Current state", ""]
    active_branch = get_active_branch(memory)
    lines.append(
        f"- **Active branch:** `{active_branch}`"
        if active_branch
        else "- **Active branch:** _(unset — create_branch records the root structure)_"
    )
    head = memory["head"]
    if head is None:
        lines.extend(["- **Head:** _(unset)_", ""])
        return lines

    current_head_state = (
        head_state(workspace, memory)
        if check_dirty and workspace is not None
        else None
    )
    suffix = (
        " _(dirty — working kernel has diverged from head)_"
        if current_head_state == "dirty"
        else ""
    )
    owner = find_branch_for_experiment(memory, head)
    lines.append(f"- **Head:** `{head}`{suffix}")
    lines.append(f"- **Head branch:** `{owner}`")
    if (
        active_branch
        and active_branch != owner
        and memory["branches"][active_branch]["base"] == head
    ):
        lines.append(
            f"- **Working position:** head is the base of active branch "
            f"`{active_branch}`."
        )
    lines.append("")
    if not expand_head:
        return lines

    target_label, target_evaluation = _target(memory)
    lines.extend(
        _render_experiment(
            head,
            memory["experiments"][head],
            target_label,
            target_evaluation,
            heading_level=3,
            marker="head variant",
        )
    )
    lines.append("")
    return lines


def _render_branch(memory: dict, branch_id: str, *, full: bool) -> list[str]:
    branch = memory["branches"][branch_id]
    parent_branch = parent_branch_id(memory, branch_id)
    branch_marker = " _(active)_" if branch_id == get_active_branch(memory) else ""
    lines = [f"### `{branch_id}`{branch_marker}"]
    lines.append(
        f"- **Parent branch:** `{parent_branch}`"
        if parent_branch
        else "- **Parent branch:** _(root)_"
    )
    lines.append(
        f"- **Base experiment:** `{branch['base']}`"
        if branch["base"]
        else "- **Base experiment:** _(none)_"
    )
    lines.append(f"- **Structure:** {branch['structure']}")

    target_label, target_evaluation = _target(memory)
    representative = branch_representative(memory, branch_id)
    if not full:
        if representative:
            evaluation = memory["experiments"][representative]["evaluation"]
            lines.append(
                f"- **Representative:** `{representative}` — "
                f"{memory['experiments'][representative]['variant']} — "
                f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
            )
            finding = memory["experiments"][representative].get("finding")
            if finding:
                lines.append(f"- **Representative finding:** {finding}")
        else:
            lines.append("- **Representative:** _(no experiment logged yet)_")
        return lines

    if branch["experiments"]:
        lines.append("- **Variants:**")
        for experiment_id in branch["experiments"]:
            marker = "head" if experiment_id == memory["head"] else ""
            rendered = _render_experiment(
                experiment_id,
                memory["experiments"][experiment_id],
                target_label,
                target_evaluation,
                heading_level=0,
                marker=marker,
            )
            lines.append(f"  - {rendered[0]}")
            lines.extend(_indent(rendered[1:], "    "))
    return lines


def _indent(lines: list[str], prefix: str) -> list[str]:
    return [prefix + line if line else line for line in lines]


def render_task_spec(memory: dict) -> list[str]:
    """Render the task-specification section used in memory Markdown."""
    spec = memory.get("task_spec") or {}
    if not spec:
        return []
    lines = ["## Task specification", ""]
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


def render_annotation(
    memory: dict,
    scope: str,
    target_id: Optional[str] = None,
) -> str:
    if scope == "hypothesis" or scope in TOP_LEVEL_SCOPES:
        return "\n".join(_render_annotation_section(memory, scope)).rstrip() + "\n"
    return render_experiment_memory(memory, target_id)


def render_tool_result(acknowledgement: str, projection: str) -> str:
    return f"{acknowledgement.rstrip()}\n\n{projection.strip()}\n"


def _experiment_order(experiment_id: str) -> tuple[int, int, str]:
    match = EXPERIMENT_RE.match(experiment_id)
    if match:
        return (0, int(match.group(1)), experiment_id)
    return (-1, -1, experiment_id)


def _branch_order(branch_id: str) -> tuple[int, str]:
    match = BRANCH_RE.match(branch_id)
    return (int(match.group(1)), branch_id) if match else (-1, branch_id)


def _memory_entry_order(entry_id: str) -> tuple[int, str]:
    match = MEMORY_ENTRY_RE.fullmatch(entry_id)
    return (int(match.group(1)), entry_id) if match else (-1, entry_id)


def _render_experiment(
    experiment_id: str,
    experiment: dict,
    target_label: Optional[str],
    target_evaluation: Optional[dict],
    *,
    heading_level: int = 4,
    marker: str = "",
) -> list[str]:
    evaluation = experiment["evaluation"]
    prefix = f"{'#' * heading_level} " if heading_level else ""
    marker_text = f" _({marker})_" if marker else ""
    lines = [f"{prefix}`{experiment_id}`{marker_text} ({evaluation['status']})"]
    lines.append(f"- **Variant:** {experiment['variant']}")
    lines.append(
        f"- **Evaluation:** "
        f"{_evaluation_summary(evaluation, target_label, target_evaluation)}"
    )
    lines.extend(
        _representative_workload_lines(
            evaluation,
            target_label,
            target_evaluation,
        )
    )
    if experiment.get("finding"):
        lines.append(f"- **Finding:** {experiment['finding']}")
    return lines


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
    over = (
        f" over {count} workload{'s' if count != 1 else ''}"
        if count
        else ""
    )
    geomean = evaluation.get("geomean_speedup_factor")
    if geomean is not None:
        parts.append(f"geomean {geomean:.2f}× vs reference{over}")
    else:
        latency = evaluation.get("geomean_latency_ms")
        if latency is not None:
            parts.append(f"geomean {_fmt_ms(latency)}{over}")
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
    representatives = evaluation["representative_workloads"]
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
            speedup = result.get("speedup_factor")
            if speedup is not None:
                metrics.append(f"{speedup:.2f}× vs reference")
            elif result.get("latency_ms") is not None:
                metrics.append(_fmt_ms(result["latency_ms"]))
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
