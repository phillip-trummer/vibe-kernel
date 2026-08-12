"""Manage hypotheses in the experiment memory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree


_ACTIONS = ("add", "replace", "remove")


SCHEMA = {
    "name": "update_memory",
    "description": (
        "Manage durable hypotheses. Adding one allocates a live id such as h0; "
        "replace and remove target that id through entry_id, never by text matching. "
        "Each hypothesis is an "
        "independently selectable prospective structural direction with a logged base; "
        "adding one does not create a branch or change the working kernel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": (
                    "Add a new hypothesis (default), replace one, or remove one."
                ),
            },
            "text": {
                "type": "string",
                "description": "Complete hypothesis. Required for add and replace.",
            },
            "base_experiment": {
                "type": "string",
                "description": (
                    "Logged experiment from which to try the hypothesis. Required "
                    "when adding or replacing hypothesis."
                ),
            },
            "entry_id": {
                "type": "string",
                "description": (
                    "Exact stable live hypothesis id shown in memory, such as h0. "
                    "Required when replacing or removing a hypothesis. Omit when "
                    "adding; the tool allocates the id."
                ),
            },
        },
    },
}


@registry.register(SCHEMA)
def update_memory(
    workspace: Path,
    action: str = "add",
    text: Optional[str] = None,
    base_experiment: Optional[str] = None,
    entry_id: Optional[str] = None,
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)

    # Validate operation
    if action not in _ACTIONS:
        return f"Error: unknown action {action!r}; expected one of: {', '.join(_ACTIONS)}."
    if action in ("add", "replace"):
        text = (text or "").strip()
        if not text:
            return f"Error: action {action!r} requires non-empty text."

    if action == "add" and entry_id is not None:
        return "Error: action 'add' allocates entry_id automatically; omit entry_id."
    if action in ("replace", "remove"):
        entry_id = (entry_id or "").strip()
        if not entry_id:
            return f"Error: action {action!r} requires entry_id."
        if not _tree.has_hypothesis(memory, entry_id):
            return (
                f"Error: hypothesis {entry_id!r} not found. "
                f"Available: {_tree.list_hypothesis_ids(memory)}"
            )
    if action in ("add", "replace"):
        if not base_experiment:
            return f"Error: action {action!r} requires base_experiment."
        if not _tree.has_experiment(memory, base_experiment):
            return (
                f"Error: experiment {base_experiment!r} not found. "
                f"Available: {_tree.list_experiment_ids(memory)}"
            )

    # Apply update
    if action == "add":
        entry_id = _tree.add_hypothesis(memory, base_experiment, text)
    elif action == "replace":
        _tree.replace_hypothesis(
            memory,
            entry_id,
            base_experiment,
            text,
        )
    else:
        _tree.consume_hypothesis(memory, entry_id)

    # Persist memory
    _tree.save_memory(workspace, memory)
    rendered = "\n".join(_tree.render_hypotheses(memory)).rstrip() + "\n"
    verb = {
        "add": "Added",
        "replace": "Replaced",
        "remove": "Removed",
    }[action]
    acknowledgement = f"{verb} hypothesis `{entry_id}`."
    return _tree.render_tool_result(acknowledgement, rendered)
