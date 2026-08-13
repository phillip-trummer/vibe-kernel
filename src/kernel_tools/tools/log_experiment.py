"""Record a fully benchmarked source snapshot in the experiment graph."""
from pathlib import Path

from kernel_tools.registry import registry

from . import _tree
from ._workspace import (
    BENCHMARK_CACHE_PATH,
    EXPERIMENTS_DIR,
    BenchmarkCache,
    read_src_files,
    solution_name_from_src_files,
)


_ACTIONS = ("advance", "fork")

SCHEMA = {
    "name": "log_experiment",
    "description": (
        "Record the fully benchmarked working source as an immutable experiment. "
        "Use action='advance' to move the active branch head forward; its previous "
        "head remains history but is no longer checkoutable. Use action='fork' to "
        "preserve the current branch head and create a new active branch from it. "
        "An empty memory accepts advance to create the root branch; fork then has no "
        "source branch and is rejected. Requires benchmark_kernel(scope='full') on "
        "this exact source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Briefly state what distinguishes this source from its parent. "
                    "Omit measurements, conclusions, history, and future plans."
                ),
            },
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "default": "advance",
                "description": (
                    "Advance the active branch, or fork a new branch while preserving "
                    "the current one."
                ),
            },
        },
        "required": ["summary"],
    },
}


@registry.register(SCHEMA)
def log_experiment(
    workspace: Path,
    summary: str,
    action: str = "advance",
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)

    summary = summary.strip()
    if not summary:
        return "Error: summary must be non-empty."
    if action not in _ACTIONS:
        return f"Error: action {action!r} must be one of: {', '.join(_ACTIONS)}."

    active_branch = _tree.get_active_branch(memory)
    if active_branch is None and action == "fork":
        return "Error: cannot fork before the root experiment has been logged."

    files = read_src_files(workspace)
    if not files:
        return "Error: the working kernel has no source files."
    solution_name = solution_name_from_src_files(files)

    cache = BenchmarkCache.load(workspace / BENCHMARK_CACHE_PATH)
    evaluation = cache.entries.get(solution_name)
    if evaluation is None:
        return (
            "Error: no full benchmark result for the current working kernel "
            f"(hashes to {solution_name!r}) — run benchmark_kernel with "
            "scope='full' first."
        )

    existing = _tree.find_experiment_by_solution(memory, solution_name)
    if existing is not None:
        return f"Error: this source snapshot is already logged as {existing}."

    parent_id = _tree.get_head(memory)
    experiment_id = (
        f"e{_tree.next_experiment_number(memory, workspace / EXPERIMENTS_DIR)}"
    )

    snapshot_dir = workspace / EXPERIMENTS_DIR / experiment_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for name, content in files:
        (snapshot_dir / name).write_text(content)

    _tree.add_experiment(
        memory,
        experiment_id=experiment_id,
        parent_id=parent_id,
        summary=summary,
        solution=solution_name,
        evaluation=evaluation.model_dump(),
    )

    if active_branch is None:
        branch_id = f"b{_tree.next_branch_number(memory)}"
        _tree.add_root_branch(memory, branch_id, experiment_id)
        acknowledgement = f"Created root branch {branch_id} at {experiment_id}."
    elif action == "advance":
        branch_id = active_branch
        _tree.advance_branch(memory, branch_id, experiment_id)
        acknowledgement = f"Advanced {branch_id} to {experiment_id}."
    else:
        branch_id = f"b{_tree.next_branch_number(memory)}"
        _tree.fork_branch(memory, branch_id, parent_id, experiment_id)
        acknowledgement = (
            f"Forked {branch_id} from {parent_id} and logged {experiment_id}."
        )

    _tree.save_memory(workspace, memory)
    return _tree.render_tool_result(
        acknowledgement,
        _tree.render_branch_projection(memory, branch_id),
    )
