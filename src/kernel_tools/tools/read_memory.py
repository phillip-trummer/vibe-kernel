"""Read an agent-sized view of the experiment memory."""
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree

SCHEMA = {
    "name": "read_memory",
    "description": (
        "Read one of two experiment-memory views. With no branch_id, return global "
        "context including all open hypotheses, the expanded head variant, and a "
        "compact catalog of every structural branch. Pass branch_id to return only "
        "that branch and all of its variants."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "branch_id": {
                "type": "string",
                "description": (
                    "Optional structural branch to inspect, e.g. "
                    "'b2_eight_stage_pipeline'."
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
