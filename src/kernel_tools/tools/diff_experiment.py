"""Diff a logged experiment against the working source or another experiment."""
import difflib
from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from . import _tree
from ._workspace import read_src_files, resolve_experiment_dir


SCHEMA = {
    "name": "diff_experiment",
    "description": (
        "Show a unified source diff from one logged experiment to the current "
        "working kernel or to another logged experiment. This is read-only and may "
        "inspect historical experiments even though only branch heads are checkout "
        "targets."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment on the left side, e.g. 'e2'.",
            },
            "against_experiment": {
                "type": "string",
                "description": (
                    "Optional logged experiment on the right side. If omitted, "
                    "compare against the working kernel."
                ),
            },
            "filename": {
                "type": "string",
                "description": "Restrict the diff to one file.",
            },
        },
        "required": ["experiment"],
    },
}


@registry.register(SCHEMA)
def diff_experiment(
    workspace: Path,
    experiment: str,
    against_experiment: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    memory = _tree.load_or_initialize_memory(workspace)
    if not _tree.has_experiment(memory, experiment):
        return (
            f"Error: experiment {experiment!r} not found. "
            f"Available: {_tree.list_experiment_ids(memory)}"
        )
    if against_experiment is not None and not _tree.has_experiment(
        memory, against_experiment
    ):
        return (
            f"Error: experiment {against_experiment!r} not found. "
            f"Available: {_tree.list_experiment_ids(memory)}"
        )

    left = _snapshot_files(workspace, experiment)
    if isinstance(left, str):
        return f"Error: {left}"
    if against_experiment is None:
        right = dict(read_src_files(workspace))
        right_label = "working_kernel"
    else:
        right = _snapshot_files(workspace, against_experiment)
        if isinstance(right, str):
            return f"Error: {right}"
        right_label = against_experiment

    if filename is not None:
        if filename not in left and filename not in right:
            return (
                f"Error: neither {experiment!r} nor {right_label!r} "
                f"contains {filename!r}."
            )
        names = [filename]
    else:
        names = sorted(left.keys() | right.keys())

    chunks = []
    for name in names:
        diff = difflib.unified_diff(
            left.get(name, "").splitlines(keepends=True),
            right.get(name, "").splitlines(keepends=True),
            fromfile=f"{experiment}/{name}",
            tofile=f"{right_label}/{name}",
        )
        text = "".join(diff)
        if text:
            chunks.append(text)

    if not chunks:
        scope = f" for {filename!r}" if filename else ""
        return f"No differences between {experiment} and {right_label}{scope}."
    return "\n".join(chunks)


def _snapshot_files(workspace: Path, experiment_id: str) -> dict[str, str] | str:
    directory = resolve_experiment_dir(workspace, experiment_id)
    if not isinstance(directory, Path):
        return directory
    return {
        path.name: path.read_text()
        for path in directory.iterdir()
        if path.is_file()
    }
