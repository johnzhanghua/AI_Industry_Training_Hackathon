import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.agent import run_financial_agent
from src.tracing import LANGSMITH_PROJECT, TRACING_ENABLED

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
def log_tracing_status():
    if TRACING_ENABLED:
        logger.info("LangSmith tracing ON -- project=%s", LANGSMITH_PROJECT)
    else:
        logger.info("LangSmith tracing OFF (set LANGSMITH_TRACING and LANGSMITH_API_KEY)")

# Input Schema for the POST request
class QueryRequest(BaseModel):
    question: str

# Output Schema matching the scoring requirements
class QueryResponse(BaseModel):
    answer: str
    steps: int
    tool_trace: List[Dict[str, Any]]

@app.get("/health")
async def health_endpoint():
    """Liveness probe. The organizer harness skips the team if this is not HTTP 200."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    # Deliberately sync, not async: run_financial_agent blocks on LiteLLM calls and
    # file reads. Declared `async def`, it would block the event loop and serialize
    # requests; as a plain `def`, FastAPI runs it in a threadpool so the documented
    # three concurrent requests are genuinely handled in parallel. All request state
    # is local to the call, so nothing is shared between concurrent invocations.
    try:
        # Call the LangGraph execution endpoint
        result = run_financial_agent(payload.question)
        
        return QueryResponse(
            answer=result["answer"],
            steps=result["steps"],
            tool_trace=result["tool_trace"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Bind on all interfaces so the organizer harness can reach the agent;
    # the README's typical setup puts the agent on port 5000.
    uvicorn.run(
        "app:app",
        host=os.getenv("AGENT_HOST", "0.0.0.0"),
        port=int(os.getenv("AGENT_PORT", "5000")),
        reload=bool(os.getenv("AGENT_RELOAD")),
    )
