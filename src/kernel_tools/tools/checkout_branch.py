"""Restore a branch head as the working kernel."""
from pathlib import Path

from kernel_tools.registry import registry

from . import _tree
from ._workspace import restore_experiment


SCHEMA = {
    "name": "checkout_branch",
    "description": (
        "Restore a preserved branch head and make that branch active. Only branch "
        "heads are checkout targets; historical experiments cannot be checked out. "
        "The checkout does not change any branch head or experiment. It is rejected "
        "when the working source differs from the current active branch head, so "
        "unlogged edits are never discarded."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch_id": {
                "type": "string",
                "description": "Branch to restore, e.g. 'b2'.",
            },
        },
        "required": ["branch_id"],
    },
}


@registry.register(SCHEMA)
def checkout_branch(workspace: Path, branch_id: str) -> str:
    memory = _tree.load_or_initialize_memory(workspace)
    if not _tree.has_branch(memory, branch_id):
        return (
            f"Error: branch {branch_id!r} not found. "
            f"Available: {_tree.list_branch_ids(memory)}"
        )

    if _tree.head_state(workspace, memory) == "dirty":
        return (
            "Error: the working source is dirty; log it or restore it manually "
            "before checking out another branch."
        )

    experiment_id = memory["branches"][branch_id]["head"]
    restored = restore_experiment(workspace, experiment_id)
    if isinstance(restored, str):
        return f"Error: {restored}"

    previous_branch = _tree.get_active_branch(memory)
    _tree.set_active_branch(memory, branch_id)
    _tree.save_memory(workspace, memory)
    acknowledgement = (
        f"Checked out {branch_id} at {experiment_id} "
        f"(active branch was {previous_branch!r}); restored {restored} file(s)."
    )
    return _tree.render_tool_result(
        acknowledgement,
        _tree.render_current_state_memory(workspace, memory),
    )
