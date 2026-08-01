"""Restore any logged experiment as the working kernel."""
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree
from ._workspace import restore_experiment


SCHEMA = {
    "name": "checkout_experiment",
    "description": (
        "Restore any logged experiment as the working kernel and make it head. "
        "Without branch_id, its owning branch becomes active. Pass a branch_id to "
        "work from that branch's base or one of its variants; this is how an existing "
        "branch can be reimplemented cleanly from its base. Use checkout to roll back, "
        "resume a structure, or select the base for a new sibling structure. "
        "Uncommitted working edits are discarded."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment_id": {
                "type": "string",
                "description": "Experiment id, e.g. 'e0_baseline' or 'e12_tiled'.",
            },
            "branch_id": {
                "type": "string",
                "description": (
                    "Optional branch to make active. The experiment must be that "
                    "branch's base or one of its variants. If omitted, the experiment's "
                    "owning branch becomes active."
                ),
            },
        },
        "required": ["experiment_id"],
    },
}


@registry.register(SCHEMA)
def checkout_experiment(
    workspace: Path,
    experiment_id: str,
    branch_id: Optional[str] = None,
) -> str:
    # Load experiment memory
    memory = _tree.load_or_initialize_memory(workspace)
    if not _tree.has_experiment(memory, experiment_id):
        return (
            f"Error: experiment {experiment_id!r} not found. "
            f"Available: {_tree.list_experiment_ids(memory)}"
        )

    # Select the structural context
    if branch_id is None:
        branch_id = _tree.find_branch_for_experiment(memory, experiment_id)
    else:
        if not _tree.has_branch(memory, branch_id):
            return (
                f"Error: branch {branch_id!r} not found. "
                f"Available: {_tree.list_branch_ids(memory)}"
            )
        if not _tree.branch_accepts_head(memory, branch_id, experiment_id):
            return (
                f"Error: experiment {experiment_id!r} is neither the base nor a "
                f"variant of branch {branch_id!r}."
            )

    # Restore snapshot
    restored = restore_experiment(workspace, experiment_id)
    if isinstance(restored, str):
        return f"Error: {restored}"

    # Advance the working cursors
    previous_head = _tree.get_head(memory)
    _tree.set_head(memory, experiment_id)
    _tree.set_active_branch(memory, branch_id)
    _tree.save_memory(workspace, memory)
    acknowledgement = (
        f"Checked out {experiment_id} (head was {previous_head!r}); restored "
        f"{restored} file(s) into the working kernel."
    )
    return _tree.render_tool_result(
        acknowledgement,
        _tree.render_current_state_memory(workspace, memory),
    )
