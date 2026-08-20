#!/usr/bin/env python3
"""Minimal agent loop: let any OpenAI-compatible chat completions model see
and use a workspace's kernel tools over MCP, standing in for a Claude Code or
Codex CLI session.

    export VIBE_KERNEL_WORKSPACE=.runs/my-run
    export OPENAI_API_KEY=...
    uv run python run_agent.py \
        --base-url https://api.swissai.svc.cscs.ch/v1 \
        --model CSCS-Inference/moonshotai/Kimi-K2.7-Code

Reads ``<workspace>/AGENTS.md`` as the system prompt and
``<workspace>/.mcp.json`` for the MCP server to spawn -- both already written
by ``scripts/configure_clients.py``, so any workspace seeded for Claude Code
or Codex works here unchanged.

Writes an append-only JSONL transcript to ``<workspace>/.agent_logs/``.

Claude models are driven through the Anthropic SDK instead of the
OpenAI-compatible path, so the conversation prefix can be cached. The loop,
the transcript format and the tools are identical either way -- only the
request shape differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client


WORKSPACE_ENV = "VIBE_KERNEL_WORKSPACE"
DEFAULT_MAX_ROUNDS = 100
DEFAULT_MAX_TOKENS = 48000
INIT_MESSAGE = "Begin kernel optimization."
CONTINUE_MESSAGE = "Continue kernel optimization."
RESET_HINT = "Your context was reset. "
MAX_RETRIES = 5
# The conversation is resent whole every round, so the prefix is worth caching.
# Rounds include a full benchmark, which routinely exceeds the 5m default TTL.
CACHE_TTL = "1h"

# Detect context overflow errors
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length",
    "reduce the length",
    "too many tokens",
    "prompt exceeds",
    "prompt is too long",
)


def _parse_arguments(text: str | None) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class TranscriptLogger:
    """Append-only JSONL transcript. One line per event, flushed immediately
    so a crash leaves a readable prefix."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self.path = path

    def write(self, **fields: Any) -> None:
        fields.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        self._file.write(json.dumps(fields, ensure_ascii=False) + "\n")
        self._file.flush()

    def message(self, message: dict[str, Any], **fields: Any) -> None:
        """Log a message verbatim as it goes onto the conversation."""
        self.write(type="message", message=message, **fields)


def _to_openai_tool(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {"type": "object", "properties": {}},
        },
    }


def _is_anthropic_model(model: str) -> bool:
    return "claude" in (model or "").lower()


def _to_anthropic_tool(tool: Any) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema or {"type": "object", "properties": {}},
    }


def _anthropic_request(messages: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """Render the canonical OpenAI-shaped history into an Anthropic request.

    Cache breakpoints are applied to copies, never to the stored history: a
    breakpoint written back into a message would change the prefix bytes on the
    next round and invalidate the very cache it was meant to create.
    """
    system: Any = None
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system = [{
                "type": "text",
                "text": message["content"],
                "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
            }]
        elif role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": message["content"]}]})
        elif role == "assistant":
            # Replay the model's own blocks verbatim, so thinking survives the round trip.
            out.append({"role": "assistant", "content": list(message["_native_content"])})
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            last = out[-1] if out else None
            if last and last["role"] == "user" and last["content"][-1].get("type") == "tool_result":
                last["content"] = [*last["content"], block]  # one user turn per tool batch
            else:
                out.append({"role": "user", "content": [block]})

    if out:
        blocks = list(out[-1]["content"])
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}
        out[-1] = {**out[-1], "content": blocks}
    return system, out


async def complete_anthropic(
    client: Any, args: argparse.Namespace, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any] | None]], dict[str, Any], str | None]:
    """One Claude turn. Streams, because max_tokens here is far past the
    non-streaming timeout."""
    system, request_messages = _anthropic_request(messages)
    kwargs: dict[str, Any] = {"tools": tools} if tools else {}
    if args.reasoning_effort:
        kwargs["output_config"] = {"effort": args.reasoning_effort}

    async with client.messages.stream(
        model=args.model, max_tokens=args.max_tokens,
        system=system, messages=request_messages, **kwargs,
    ) as stream:
        response = await stream.get_final_message()

    native = [block.model_dump(exclude_none=True) for block in response.content]
    text = "".join(block.text for block in response.content if block.type == "text")
    calls = [
        (block.id, block.name, dict(block.input or {}))
        for block in response.content
        if block.type == "tool_use"
    ]
    message = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}
            for cid, name, arguments in calls
        ],
        "_native_content": native,
    }
    return message, calls, response.usage.model_dump(), response.stop_reason


async def complete_openai(
    client: Any, args: argparse.Namespace, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any] | None]], dict[str, Any], str | None]:
    """One turn against any OpenAI-compatible backend."""
    kwargs: dict[str, Any] = {"tools": tools} if tools else {}
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort

    response = await client.chat.completions.create(
        model=args.model, messages=[_strip_native(m) for m in messages],
        max_tokens=args.max_tokens, **kwargs,
    )
    choice = response.choices[0]
    calls = [
        (call.id, call.function.name, _parse_arguments(call.function.arguments))
        for call in choice.message.tool_calls or []
    ]
    return (
        choice.message.model_dump(exclude_none=True),
        calls,
        response.usage.model_dump() if response.usage else {},
        choice.finish_reason,
    )


def _strip_native(message: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in message.items() if k != "_native_content"}


async def call_mcp_tool(
    session: Client, name: str, arguments: dict[str, Any]
) -> tuple[str, bool]:
    try:
        result = await session.call_tool(name, arguments)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", True
    text = "\n".join(block.text for block in result.content if block.type == "text")
    return text, result.is_error


def load_workspace(workspace: Path) -> tuple[str, str, StdioServerParameters]:
    """Read the system prompt and MCP server config that
    ``scripts/configure_clients.py`` wrote into a workspace."""
    agents_md = workspace / "AGENTS.md"
    mcp_json = workspace / ".mcp.json"
    for path in (agents_md, mcp_json):
        if not path.is_file():
            raise SystemExit(f"Error: {path} not found; run scripts/configure_clients.py first.")

    mcp_config = json.loads(mcp_json.read_text())
    server_name, server_cfg = next(iter(mcp_config["mcpServers"].items()))
    params = StdioServerParameters(command=server_cfg["command"], args=server_cfg.get("args", []))
    return agents_md.read_text(), server_name, params


async def run(args: argparse.Namespace) -> None:
    """Connect to the workspace's MCP server and run the agent loop until
    max_roundss."""
    workspace = args.workspace.expanduser().resolve()
    system_prompt, server_name, params = load_workspace(workspace)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TranscriptLogger(workspace / ".agent_logs" / f"{run_id}.jsonl")
    print(f"[Transcript] {logger.path}")

    # Spawn the MCP server and collect its tools.
    async with Client(stdio_client(params)) as session:
        mcp_tools = (await session.list_tools()).tools
        tools = [_to_openai_tool(tool) for tool in mcp_tools]
        anthropic_tools = [_to_anthropic_tool(tool) for tool in mcp_tools]
        print(f"[MCP] connected to {server_name}; {len(tools)} tool(s) available")

        logger.write(
            type="meta",
            base_url=args.base_url,
            model=args.model,
            server=server_name,
            tool_count=len(tools),
        )

        # Initialize conversation.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": INIT_MESSAGE},
        ]
        for msg in messages:
            logger.message(msg)

        use_anthropic = _is_anthropic_model(args.model)
        if use_anthropic:
            import anthropic

            complete = complete_anthropic
            client_cm = anthropic.AsyncAnthropic(
                base_url=args.base_url,
                api_key=args.api_key,
                max_retries=MAX_RETRIES,
                timeout=args.request_timeout,
            )
            request_tools = anthropic_tools
        else:
            complete = complete_openai
            client_cm = openai.AsyncOpenAI(
                base_url=args.base_url,
                api_key=args.api_key,
                max_retries=MAX_RETRIES,
                timeout=args.request_timeout,
            )
            request_tools = tools
        print(f"[Model] {args.model} via {'anthropic' if use_anthropic else 'openai'} SDK"
              + (f"; prompt caching on ({CACHE_TTL} TTL)" if use_anthropic else ""))

        async with client_cm as client:
            for round_idx in range(args.max_rounds):
                print(f"\n=== Round {round_idx + 1}/{args.max_rounds} ===")
                start = time.perf_counter()
                # Ask agent.
                try:
                    message, calls, usage, finish_reason = await complete(
                        client, args, messages, request_tools
                    )
                except Exception as e:
                    if not any(marker in str(e).lower() for marker in _CONTEXT_OVERFLOW_MARKERS):
                        logger.write(type="error", error=f"{type(e).__name__}: {e}")
                        raise
                    # Reset the conversation.
                    print("[Context window full -- resetting conversation]")
                    logger.write(
                        type="context_reset", error=f"{type(e).__name__}: {e}"
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": RESET_HINT + CONTINUE_MESSAGE},
                    ]
                    for msg in messages:
                        logger.message(msg)
                    continue
                elapsed = time.perf_counter() - start

                text = message.get("content") or ""
                messages.append(message)
                logger.message(
                    _strip_native(messages[-1]),
                    finish_reason=finish_reason,
                    usage=usage,
                    elapsed_s=elapsed,
                )
                if text:
                    print(text)

                # The agent stopped without calling a tool. The CLI protocol
                # answers this with /clear + "continue", so a stop is a context
                # boundary; --on-stop nudge keeps the conversation instead.
                if not calls:
                    logger.write(type="agent_stop")
                    if args.on_stop == "clear":
                        print("[Agent stopped; clearing context and continuing]")
                        logger.write(type="context_reset", reason="agent_stop")
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": CONTINUE_MESSAGE},
                        ]
                        for msg in messages:
                            logger.message(msg)
                    else:
                        print("[Agent stopped; nudging to continue]")
                        messages.append({"role": "user", "content": CONTINUE_MESSAGE})
                        logger.message(messages[-1])
                    continue

                # Run requested tools.
                for call_id, name, arguments in calls:
                    print(f"Tool call: [{name}] {str(arguments)[:200]}")
                    if arguments is None:
                        output, is_error = "Error: arguments were not a JSON object.", True
                    else:
                        output, is_error = await call_mcp_tool(session, name, arguments)
                    if is_error:
                        print(f"Tool error: {output[:500]}")

                    # Send tool result back.
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": output}
                    )
                    logger.message(messages[-1], name=name, is_error=is_error)
            print(f"[stopped after max_rounds={args.max_rounds}]")

        logger.write(type="agent_stop")

    print(f"[Done] transcript: {logger.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    workspace_env = os.environ.get(WORKSPACE_ENV)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(workspace_env) if workspace_env else None,
        required=not workspace_env,
        help=f"Workspace path (default: ${WORKSPACE_ENV} when set).",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL, e.g. https://api.openai.com/v1 "
        "(default: $OPENAI_BASE_URL, resolved by the OpenAI SDK).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL"),
        required="OPENAI_MODEL" not in os.environ,
        help="Backend model name (default: $OPENAI_MODEL).",
    )
    parser.add_argument(
        "--api-key",
        help="API key (default: $OPENAI_API_KEY, resolved by the OpenAI SDK).",
    )
    parser.add_argument(
        "--on-stop",
        choices=("clear", "nudge"),
        default="clear",
        help="What to do when the agent stops without calling a tool: 'clear' "
        "discards the conversation and continues (the CLI protocol: /clear then "
        "\"continue kernel optimization\"), 'nudge' asks it to continue in the "
        "same conversation (default: clear).",
    )
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT"),
        help="Backend-specific reasoning/thinking level, e.g. low/medium/high "
        "(default: $OPENAI_REASONING_EFFORT; omitted from the request if unset).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-request HTTP timeout in seconds (default: 1800). Must cover "
        "--max-tokens at the backend's generation rate plus queueing.",
    )
    args = parser.parse_args(argv)

    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
