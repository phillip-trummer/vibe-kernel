"""Update durable knowledge in the experiment memory."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree


_ACTIONS = ("add", "replace", "remove")
_SCOPES = ("hypothesis",) + _tree.TOP_LEVEL_SCOPES + _tree.FINDING_SCOPES


SCHEMA = {
    "name": "update_memory",
    "description": (
        "Revise concise durable knowledge using exact stable ids. Adding a hypothesis, "
        "fact, or hazard allocates a live id such as h0, f0, or r0; replace and remove "
        "target that id through entry_id, never by text matching. Each hypothesis is "
        "an independently selectable prospective structural direction with a logged "
        "base; adding one does not create a branch or change the working kernel. Facts "
        "and hazards are terse cross-branch statements. An experiment finding is one "
        "causal lesson, not a running note or measurement summary, and is addressed by "
        "its experiment_id. 'add' creates, 'replace' revises, and 'remove' deletes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": list(_SCOPES),
                "description": "The durable memory field to revise.",
            },
            "action": {
                "type": "string",
                "enum": list(_ACTIONS),
                "description": "Add a new entry (default), replace one, or remove one.",
            },
            "text": {
                "type": "string",
                "description": (
                    "Complete concise statement to store. Required for add and replace."
                ),
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
                    "Exact stable live id shown in memory, such as h0, f0, or r0. "
                    "Required when replacing or removing a hypothesis, fact, or hazard. "
                    "Omit when adding; the tool allocates the id."
                ),
            },
            "experiment_id": {
                "type": "string",
                "description": "Required when scope is experiment_finding.",
            },
        },
        "required": ["scope"],
    },
}


@registry.register(SCHEMA)
def update_memory(
    workspace: Path,
    scope: str,
    action: str = "add",
    text: Optional[str] = None,
    base_experiment: Optional[str] = None,
    entry_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)

    # Validate operation
    if scope not in _SCOPES:
        return f"Error: unknown scope {scope!r}; expected one of: {', '.join(_SCOPES)}."
    if action not in _ACTIONS:
        return f"Error: unknown action {action!r}; expected one of: {', '.join(_ACTIONS)}."
    if action in ("add", "replace"):
        text = (text or "").strip()
        if not text:
            return f"Error: action {action!r} requires non-empty text."

    # Resolve target
    target_id = None
    if scope == "hypothesis" or scope in _tree.TOP_LEVEL_SCOPES:
        if action == "add" and entry_id is not None:
            return (
                "Error: action 'add' allocates entry_id automatically; omit entry_id."
            )
        if action in ("replace", "remove"):
            entry_id = (entry_id or "").strip()
            if not entry_id:
                return f"Error: action {action!r} for {scope} requires entry_id."
            if not _tree.has_memory_entry(memory, scope, entry_id):
                return (
                    f"Error: {scope} {entry_id!r} not found. "
                    f"Available: {_tree.list_memory_entry_ids(memory, scope)}"
                )
    if scope == "hypothesis":
        if action in ("add", "replace"):
            if not base_experiment:
                return (
                    f"Error: action {action!r} for hypothesis requires "
                    "base_experiment."
                )
            if not _tree.has_experiment(memory, base_experiment):
                return (
                    f"Error: experiment {base_experiment!r} not found. "
                    f"Available: {_tree.list_experiment_ids(memory)}"
                )
    if scope == "experiment_finding":
        target_id = (experiment_id or "").strip()
        if not target_id:
            return "Error: scope 'experiment_finding' requires experiment_id."
        if not _tree.has_experiment(memory, target_id):
            return (
                f"Error: experiment {target_id!r} not found. "
                f"Available: {_tree.list_experiment_ids(memory)}"
            )

    # Apply hypothesis update
    if scope == "hypothesis":
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

    # Apply fact or hazard update
    elif scope in _tree.TOP_LEVEL_SCOPES:
        if action == "add":
            entry_id = _tree.add_annotation(memory, scope, text)
        elif action == "replace":
            _tree.replace_annotation(memory, scope, entry_id, text)
        else:
            _tree.remove_annotation(memory, scope, entry_id)

    # Apply finding update
    else:
        current = _tree.get_finding(memory, scope, target_id)
        if action == "add":
            if current:
                return (
                    f"Error: {scope.replace('_', ' ')} already exists; "
                    "use replace to revise it."
                )
            _tree.set_finding(memory, scope, target_id, text)
        elif action == "replace":
            if not current:
                return (
                    f"Error: experiment {target_id!r} has no finding to replace; "
                    "use add."
                )
            _tree.set_finding(memory, scope, target_id, text)
        else:
            if not current:
                return (
                    f"Error: experiment {target_id!r} has no finding to remove."
                )
            _tree.set_finding(memory, scope, target_id, None)

    # Persist memory
    _tree.save_memory(workspace, memory)
    rendered = _tree.render_annotation(memory, scope, target_id)
    verb = {
        "add": "Added",
        "replace": "Replaced",
        "remove": "Removed",
    }[action]
    if scope == "experiment_finding":
        acknowledgement = f"{verb} experiment finding for `{target_id}`."
    else:
        acknowledgement = f"{verb} {scope} `{entry_id}`."
    return _tree.render_tool_result(acknowledgement, rendered)
