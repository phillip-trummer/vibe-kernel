"""Read an agent-sized view of the experiment memory."""
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree

SCHEMA = {
    "name": "read_memory",
    "description": (
        "Read experiment memory. With no branch_id, return task and global results, "
        "the active branch's clean or dirty state and local history, every other "
        "branch's head, and ideas grouped by branch. Pass branch_id to expand only "
        "that branch's local history and ideas. Reading never changes the active "
        "branch or the workspace markdown file."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch_id": {
                "type": "string",
                "description": (
                    "Optional branch to inspect, e.g. 'b2'."
                ),
            },
        },
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
