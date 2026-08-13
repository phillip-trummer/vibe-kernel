"""Add or remove branch-associated ideas."""
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree


_ACTIONS = ("add", "remove")
_MAX_IDEA_LENGTH = 300

SCHEMA = {
    "name": "update_idea",
    "description": (
        "Add an optional idea for the active branch, or remove an idea by id. "
        "Each idea is one concrete, not-yet-tried direction for future work. Do not "
        "use ideas to record completed attempts, findings, profiling interpretations, "
        "or conclusions. "
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "default": "add",
                "description": "Add a new idea (default), or remove one.",
            },
            "text": {
                "type": "string",
                "maxLength": _MAX_IDEA_LENGTH,
                "description": (
                    "One concise, concrete direction that has not yet been tried. "
                    "Do not include past results or conclusions. Required for add; "
                    f"maximum {_MAX_IDEA_LENGTH} characters."
                ),
            },
            "idea_id": {
                "type": "string",
                "description": "Idea id to remove, e.g. 'i3'. Required for remove.",
            },
        },
    },
}


@registry.register(SCHEMA)
def update_idea(
    workspace: Path,
    action: str = "add",
    text: Optional[str] = None,
    idea_id: Optional[str] = None,
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)
    if action not in _ACTIONS:
        return f"Error: action {action!r} must be one of: {', '.join(_ACTIONS)}."

    if action == "add":
        branch_id = _tree.get_active_branch(memory)
        if branch_id is None:
            return "Error: log the root experiment before adding an idea."
        text = (text or "").strip()
        if not text:
            return "Error: action 'add' requires non-empty text."
        if len(text) > _MAX_IDEA_LENGTH:
            return (
                f"Error: idea text must be at most {_MAX_IDEA_LENGTH} characters "
                f"(received {len(text)})."
            )
        if idea_id is not None:
            return "Error: action 'add' allocates idea_id automatically; omit idea_id."
        idea_id = _tree.add_idea(memory, branch_id, text)
        acknowledgement = f"Added idea `{idea_id}` to `{branch_id}`."
    else:
        idea_id = (idea_id or "").strip()
        if not idea_id:
            return "Error: action 'remove' requires idea_id."
        if not _tree.has_idea(memory, idea_id):
            return (
                f"Error: idea {idea_id!r} not found. "
                f"Available: {_tree.list_idea_ids(memory)}"
            )
        branch_id = memory["ideas"][idea_id]["branch_id"]
        _tree.remove_idea(memory, idea_id)
        acknowledgement = f"Removed idea `{idea_id}` from `{branch_id}`."

    _tree.save_memory(workspace, memory)
    projection = "\n".join(_tree.render_ideas(memory, branch_id)).rstrip() + "\n"
    return _tree.render_tool_result(acknowledgement, projection)
