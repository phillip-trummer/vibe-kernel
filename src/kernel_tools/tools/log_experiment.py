"""Record a fully benchmarked tuning experiment."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree
from ._workspace import (
    BENCHMARK_CACHE_PATH,
    EXPERIMENTS_DIR,
    BenchmarkCache,
    read_src_files,
    solution_name_from_src_files,
)


SCHEMA = {
    "name": "log_experiment",
    "description": (
        "Record the fully benchmarked working kernel as another variant of the "
        "active structural branch. Head must be either that branch's base or one "
        "of its variants. Use create_branch instead when the experiment establishes "
        "a distinct structural direction worth preserving and revisiting independently "
        "from its base. Benchmark outcome does not decide this classification. Log "
        "every informative variant, including regressions and correctness failures. "
        "Requires benchmark_kernel(scope='full') on this exact source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Short experiment identifier using lowercase letters, digits, "
                    "and underscores. Do not include the assigned eN prefix."
                ),
            },
            "variant": {
                "type": "string",
                "description": (
                    "Concise self-contained description of this concrete realization "
                    "of the branch structure. State the parameter, schedule, layout, "
                    "synchronization, or implementation choices that distinguish it. "
                    "Describe the resulting implementation, not a transition from "
                    "another experiment; omit measurements and findings."
                ),
            },
            "finding": {
                "type": "string",
                "description": (
                    "Optional non-obvious lesson or failure diagnosis needed by "
                    "future work. Omit it when the recorded evaluation speaks for itself."
                ),
            },
        },
        "required": ["slug", "variant"],
    },
}

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


@registry.register(SCHEMA)
def log_experiment(
    workspace: Path,
    slug: str,
    variant: str,
    finding: Optional[str] = None,
) -> str:
    return _record_experiment(
        workspace,
        slug=slug,
        variant=variant,
        finding=finding,
        structure=None,
        hypothesis_id=None,
    )


def _record_experiment(
    workspace: Path,
    *,
    slug: str,
    variant: str,
    finding: Optional[str],
    structure: Optional[str],
    hypothesis_id: Optional[str],
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)

    # Validate experiment
    if not _SLUG_RE.fullmatch(slug):
        return (
            f"Error: slug {slug!r} must match {_SLUG_RE.pattern} "
            "(lowercase letters, digits, underscores only)."
        )
    variant = variant.strip()
    if not variant:
        return "Error: variant must be non-empty."
    finding = (finding or "").strip() or None
    if structure is not None:
        structure = structure.strip()
        if not structure:
            return "Error: structure must be non-empty."
    if hypothesis_id is not None:
        hypothesis_id = hypothesis_id.strip()
        if not hypothesis_id:
            return "Error: hypothesis_id must be non-empty when provided."
        if structure is None:
            return "Error: only create_branch can consume a hypothesis."

    # Identify working kernel
    files = read_src_files(workspace)
    if not files:
        return "Error: the working kernel has no source files."
    solution_name = solution_name_from_src_files(files)

    # Load full evaluation
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

    # Resolve structural branch
    head = _tree.get_head(memory)
    if hypothesis_id is not None:
        if not _tree.has_hypothesis(memory, hypothesis_id):
            return (
                f"Error: hypothesis {hypothesis_id!r} not found. "
                f"Available: {_tree.list_hypothesis_ids(memory)}"
            )
        hypothesis = _tree.get_hypothesis(memory, hypothesis_id)
        if hypothesis["base"] != head:
            return (
                f"Error: hypothesis {hypothesis_id!r} is based on "
                f"{hypothesis['base']!r}, but head is {head!r}; check out the "
                "hypothesis base before creating its branch."
            )
    bare_slug = re.sub(r"^[eb]\d+_", "", slug)
    if structure is not None:
        branch_id = f"b{_tree.next_branch_number(memory)}_{bare_slug}"
    else:
        branch_id = _tree.get_active_branch(memory)
        if branch_id is None:
            return (
                "Error: no active structural branch — use create_branch to record "
                "the first implemented structure."
            )
        if not _tree.branch_accepts_head(memory, branch_id, head):
            return (
                f"Error: head {head!r} is neither the base nor a variant of active "
                f"branch {branch_id!r}; check out an allowed experiment and select "
                "the intended branch first."
            )

    # Allocate experiment
    experiment_id = (
        f"e{_tree.next_experiment_number(memory, workspace / EXPERIMENTS_DIR)}_{bare_slug}"
    )

    # Snapshot sources
    snapshot_dir = workspace / EXPERIMENTS_DIR / experiment_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files:
        (snapshot_dir / name).write_text(content)

    # Record experiment
    evaluation_dict = evaluation.model_dump()
    if structure is not None:
        _tree.add_branch(
            memory,
            branch_id=branch_id,
            base=head,
            structure=structure,
        )
    _tree.add_experiment(
        memory,
        experiment_id=experiment_id,
        branch_id=branch_id,
        solution=solution_name,
        variant=variant,
        finding=finding,
        evaluation=evaluation_dict,
    )
    if hypothesis_id is not None:
        _tree.consume_hypothesis(memory, hypothesis_id)

    # Advance the working cursors
    _tree.set_head(memory, experiment_id)
    _tree.set_active_branch(memory, branch_id)

    # Persist memory
    _tree.save_memory(workspace, memory)
    branch_created = structure is not None
    acknowledgement = (
        f"Created branch {branch_id} and logged {experiment_id}."
        if branch_created
        else f"Logged {experiment_id} on branch {branch_id}."
    )
    if hypothesis_id is not None:
        acknowledgement += f"\nConsumed hypothesis `{hypothesis_id}`."
    projection = (
        _tree.render_branch_projection(memory, branch_id)
        if branch_created
        else _tree.render_experiment_memory(memory, experiment_id)
    )
    return _tree.render_tool_result(acknowledgement, projection)
