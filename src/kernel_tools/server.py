"""MCP server and stdio transport."""

from __future__ import annotations

from pathlib import Path

import mcp.types as types
from mcp import MCPError
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server

from kernel_tools.registry import ToolNotAvailableError, ToolRegistry


def text_result(text: str, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        is_error=is_error,
    )


def create_server(
    *,
    workspace: Path,
    registry: ToolRegistry,
) -> Server:
    async def list_tools(
        ctx: ServerRequestContext,
        params: types.PaginatedRequestParams | None,
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=registry.tools)

    async def call_tool(
        ctx: ServerRequestContext,
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        try:
            result = registry.call(
                params.name,
                workspace,
                params.arguments or {},
            )
        except ToolNotAvailableError as e:
            raise MCPError(types.INVALID_PARAMS, str(e)) from e
        except Exception as e:
            return text_result(f"Error: {e}", is_error=True)

        if isinstance(result, types.CallToolResult):
            return result
        text = str(result)
        return text_result(text, is_error=text.startswith("Error:"))

    return Server(
        "kernel-tools-mcp",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        # Close the write stream so the process exits when stdin closes.
        async with write_stream:
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
