"""Tool registration and allowlist selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Any

from mcp.types import Tool


class ToolNotAvailableError(ValueError):
    pass


class ToolRegistry:
    def __init__(
        self,
        entries: dict[str, tuple[Tool, Callable[..., Any]]] | None = None,
    ):
        self._entries = entries or {}

    @property
    def tools(self) -> list[Tool]:
        return [tool for tool, _ in self._entries.values()]

    def register(self, schema: dict[str, Any]):
        tool = Tool(**schema)

        def decorator(handler: Callable[..., Any]):
            if tool.name in self._entries:
                raise ValueError(f"duplicate tool: {tool.name}")
            self._entries[tool.name] = (tool, handler)
            return handler

        return decorator

    def select(self, names: Iterable[str] | None) -> ToolRegistry:
        if names is None:
            return self

        selected = set(names)
        unknown = selected - self._entries.keys()
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")

        entries = {
            name: entry
            for name, entry in self._entries.items()
            if name in selected
        }
        return ToolRegistry(entries)

    def call(
        self,
        name: str,
        workspace: Path,
        arguments: dict[str, Any],
    ) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise ToolNotAvailableError(f"tool is not available: {name}")

        return entry[1](workspace, **arguments)


registry = ToolRegistry()


def load_registry() -> ToolRegistry:
    import_module("kernel_tools.tools")
    return registry
