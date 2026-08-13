"""Add optional Codex and Claude Code configuration to a workspace.

    uv run python scripts/configure_clients.py \
        --workspace .runs/mla-sol \
        --template simple.md \
        --deny-builtins
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CLAUDE_INSTRUCTIONS = "@AGENTS.md\n"
REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "templates"
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAMES = (
    "read_source",
    "write_source",
    "edit_source",
    "benchmark_kernel",
    "benchmark_sweep",
    "profile_kernel",
    "read_memory",
    "log_experiment",
    "checkout_branch",
    "diff_experiment",
    "update_idea",
)
MCP_TOOL_TIMEOUT_SECONDS = 4200


def configure_clients(
    workspace: Path,
    template: str,
    force: bool = False,
    tools: list[str] | tuple[str, ...] | None = None,
    deny_builtins: bool = False,
    include_task_description: bool = False,
) -> None:
    workspace = workspace.expanduser().resolve()
    missing = [
        name for name in ("task", "src") if not (workspace / name).is_dir()
    ]
    if missing:
        raise SystemExit(
            f"Error: workspace is missing: {', '.join(missing)}"
        )

    if Path(template).name != template:
        raise SystemExit("Error: --template must be a filename from templates/.")
    template_path = TEMPLATE_DIR / template
    if not template_path.is_file():
        available = ", ".join(
            path.name for path in sorted(TEMPLATE_DIR.glob("*.md"))
        )
        raise SystemExit(
            f"Error: unknown template {template!r}. Available: {available}"
        )

    tool_names = TOOL_NAMES if tools is None else tuple(tools)
    if not tool_names:
        raise SystemExit("Error: at least one tool is required.")
    unknown = set(tool_names) - set(TOOL_NAMES)
    if unknown:
        raise SystemExit(f"Error: unknown tools: {', '.join(sorted(unknown))}")

    server_args = [
        "kernel-tools-mcp",
        "--workspace",
        str(workspace),
        "--tools",
        *tool_names,
    ]
    mcp_config = {
        "mcpServers": {
            "kernel-tools": {
                "type": "stdio",
                "command": "uv",
                "args": ["run", "--project", str(REPO_ROOT), *server_args],
                "timeout": MCP_TOOL_TIMEOUT_SECONDS * 1000,
            }
        }
    }
    codex_features = (
        "[features]\n"
        "shell_tool = false\n\n"
        if deny_builtins
        else ""
    )
    codex_config = (
        codex_features
        + "[mcp_servers.kernel-tools]\n"
        'command = "uv"\n'
        f"cwd = {json.dumps(str(REPO_ROOT))}\n"
        f"args = {json.dumps(['run', *server_args])}\n"
        f"enabled_tools = {json.dumps(list(tool_names))}\n"
        'default_tools_approval_mode = "approve"\n'
        f"tool_timeout_sec = {MCP_TOOL_TIMEOUT_SECONDS}\n"
    )
    permissions = {
        "allow": [f"mcp__kernel-tools__{name}" for name in tool_names],
    }
    if deny_builtins:
        permissions["deny"] = ["Bash", "Read", "Edit"]

    claude_settings = {
        "$schema": "https://json.schemastore.org/claude-code-settings.json",
        "enabledMcpjsonServers": ["kernel-tools"],
        "permissions": permissions,
    }
    instructions = template_path.read_text()
    if include_task_description:
        from kernel_tools.tools import _tree

        try:
            memory = _tree.create_memory(workspace)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"Error: could not read workspace task: {exc}"
            ) from exc
        task_specification = "\n".join(
            _tree.render_task_spec(memory)
        ).rstrip()
        if task_specification:
            instructions = (
                f"{instructions.rstrip()}\n\n{task_specification}\n"
            )

    files = {
        workspace / "AGENTS.md": instructions,
        workspace / "CLAUDE.md": CLAUDE_INSTRUCTIONS,
        workspace / ".mcp.json": json.dumps(mcp_config, indent=2) + "\n",
        workspace / ".claude" / "settings.json": (
            json.dumps(claude_settings, indent=2) + "\n"
        ),
        workspace / ".codex" / "config.toml": codex_config,
    }
    existing = [path for path in files if path.exists()]
    if existing and not force:
        names = ", ".join(str(path.relative_to(workspace)) for path in existing)
        raise SystemExit(f"Error: refusing to replace existing files: {names}")

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    print(f"Configured agent clients in {workspace}")
    print("Claude Code will load the project-scoped kernel-tools MCP server.")
    print("Codex will load the project-scoped kernel-tools MCP server.")
    print(f"Instructions: {template_path.name}")
    if include_task_description:
        print("Included task specification in AGENTS.md")
    print(f"Allowed tools: {', '.join(tool_names)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--template",
        required=True,
        help="Instruction filename from templates/.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        help="Tools to expose and allow; defaults to all tools.",
    )
    parser.add_argument(
        "--deny-builtins",
        action="store_true",
        help=(
            "Deny Claude Code's Bash, Read, and Edit tools, and disable "
            "Codex's shell tool."
        ),
    )
    parser.add_argument(
        "--include-task-description",
        action="store_true",
        help="Append the complete task specification to AGENTS.md.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing instructions, .mcp.json, and Claude settings.",
    )
    args = parser.parse_args(argv)
    configure_clients(
        args.workspace,
        args.template,
        force=args.force,
        tools=args.tools,
        deny_builtins=args.deny_builtins,
        include_task_description=args.include_task_description,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
