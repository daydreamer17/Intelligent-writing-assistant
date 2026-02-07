from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...models.schemas import MCPToolCallRequest, MCPToolCallResponse, MCPToolsResponse
from ..deps import AppServices, get_services

router = APIRouter(tags=["mcp"])
logger = logging.getLogger("app.mcp")


@router.get("/mcp/github/tools", response_model=MCPToolsResponse)
def list_github_tools(services: AppServices = Depends(get_services)) -> MCPToolsResponse:
    tool = services.github_mcp_tool
    if not tool:
        raise HTTPException(status_code=503, detail="GitHub MCP not enabled")

    tools = []
    try:
        available = getattr(tool, "_available_tools", []) or []
        tools = [
            {"name": t.get("name", ""), "description": t.get("description", "")}
            for t in available
        ]
    except Exception:
        tools = []

    raw = ""
    if not tools:
        try:
            raw = tool.run({"action": "list_tools"})
        except Exception as exc:
            logger.warning("Failed to list GitHub MCP tools: %s", exc)

    return MCPToolsResponse(tools=tools, raw=raw)


@router.post("/mcp/github/call", response_model=MCPToolCallResponse)
def call_github_tool(
    payload: MCPToolCallRequest,
    services: AppServices = Depends(get_services),
) -> MCPToolCallResponse:
    tool = services.github_mcp_tool
    if not tool:
        raise HTTPException(status_code=503, detail="GitHub MCP not enabled")

    try:
        result = tool.run(
            {
                "action": "call_tool",
                "tool_name": payload.tool_name,
                "arguments": payload.arguments,
            }
        )
        return MCPToolCallResponse(result=result)
    except Exception as exc:
        logger.exception("GitHub MCP tool call failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
