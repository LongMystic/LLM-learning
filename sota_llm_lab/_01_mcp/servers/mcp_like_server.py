from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from tools import call_tool

class ToolRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    ok: bool
    result: Any = None
    error: str | None = None


app = FastAPI(title="MCP-like Tool Server")

@app.post("/call_tool", response_model=ToolResponse)
def call_tool_response(payload: ToolRequest):
    try:
        result = call_tool(payload.tool, payload.args)
        return ToolResponse(ok=True, result=result)
    except Exception as e:
        return ToolResponse(ok=True, error=str(e))