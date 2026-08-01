"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from kernel_tools.workspace import resolve_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-tools-mcp")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace root; defaults to the current directory.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        metavar="NAME",
        help="Tools to expose; defaults to all tools.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Load tools only when starting the server.
    from kernel_tools.registry import load_registry
    from kernel_tools.server import create_server, serve_stdio

    try:
        workspace = resolve_workspace(args.workspace)
        active_tools = load_registry().select(args.tools)
    except ValueError as e:
        print(f"kernel-tools-mcp: error: {e}", file=sys.stderr)
        return 2

    server = create_server(workspace=workspace, registry=active_tools)
    asyncio.run(serve_stdio(server))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
