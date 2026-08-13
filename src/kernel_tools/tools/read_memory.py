"""Read an agent-sized view of the experiment memory."""
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree

SCHEMA = {
    "name": "read_memory",
    "description": (
        "Review experiment memory. Omit branch_id for an overview of the current "
        "state, recorded results, available branches, and ideas. Pass branch_id to "
        "inspect one branch's history in detail when evaluating where to continue. "
        "Reading memory never switches branches or restores source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch_id": {
                "type": "string",
                "pattern": r"^b[0-9]+$",
                "description": (
                    "Branch to inspect, e.g. 'b2'. Omit for the overview."
                ),
            },
        },
        "additionalProperties": False,
    },
}


@registry.register(SCHEMA)
def read_memory(workspace: Path, branch_id: Optional[str] = None) -> str:
    # Load memory
    memory = _tree.load_or_initialize_memory(workspace)

    # Render catalog
    if branch_id is None:
        return _tree.render_catalog_memory(workspace, memory)

    # Render branch
    if not _tree.has_branch(memory, branch_id):
        return (
            f"Error: branch {branch_id!r} not found. "
            f"Available: {_tree.list_branch_ids(memory)}"
        )
    return _tree.render_branch_memory(memory, branch_id)
