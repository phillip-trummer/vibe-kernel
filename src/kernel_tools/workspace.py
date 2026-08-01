"""Workspace validation."""

from pathlib import Path


class WorkspaceError(ValueError):
    pass


def resolve_workspace(path: Path | str | None = None) -> Path:
    workspace = Path.cwd() if path is None else Path(path).expanduser()
    workspace = workspace.resolve()

    if not workspace.is_dir():
        raise WorkspaceError(f"workspace does not exist: {workspace}")

    missing = [
        name
        for name in ("task", "src")
        if not (workspace / name).is_dir()
    ]
    if missing:
        raise WorkspaceError(
            f"workspace is missing: {', '.join(missing)}"
        )

    return workspace
