import os
import uuid
import logging # Import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # Allow frontend calls
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import asyncio
import json
from fastapi.responses import HTMLResponse, StreamingResponse

from langchain_core.messages import HumanMessage

from .graph import app # Import the compiled LangGraph
from .state import AgentState # Import state definition if needed for input/output models

# --- Logging Configuration --- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
fastapi_app = FastAPI(
    title="LangGraph Multi-Agent Research Report API",
    description="API for generating research reports using a multi-agent LangGraph workflow.",
    version="1.0.0"
)

# --- CORS Middleware --- #
# Allow requests from your Streamlit frontend (adjust origins if needed)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
    expose_headers=["*"]
)

# --- Request and Response Models --- #
class GenerateRequest(BaseModel):
    prompt: str

# No longer using GenerateResponse, as we stream
# class GenerateResponse(BaseModel):
#     final_report: str | None
#     error: str | None

# --- Streaming Generator Function --- #
async def stream_graph_events(prompt: str, thread_id: str):
    """Runs the graph and yields formatted Server-Sent Events."""
    initial_state = {
        "user_prompt": prompt,
        "messages": [HumanMessage(content=prompt)],
        "research_findings": [],
        "plan": None,
        "draft_report": None,
        "final_report": None,
        "review_feedback": None,
        "next_agent": None,
        "revision_count": 0,
    }

    final_report = None
    error_message = None
    current_step = 1

    try:
        async for event in app.astream(initial_state):
            if isinstance(event, dict):
                # Determine the current node/step based on the keys
                node_name = list(event.keys())[0] # Get the primary key indicating the node
                node_data = event[node_name]
                
                # Prepare the data payload for the SSE event
                update_data = {
                    "type": "update",
                    "step": current_step,
                    "node": node_name,
                    "data": {}
                }
                
                # Extract useful info from the node data
                if node_name == "planner" and node_data.get("plan"):
                    update_data["data"]["summary"] = f"Planner created plan: {node_data['plan'][:100]}..."
                elif "researcher" in node_name and node_data.get("research_findings"):
                    last_finding = node_data["research_findings"][-1]
                    update_data["data"]["summary"] = f"{last_finding.get('agent_name', node_name)} found research on '{last_finding.get('topic')}': {last_finding.get('research', '')[:100]}..."
                elif node_name == "writer" and node_data.get("draft_report"):
                    rev_count = node_data.get("revision_count", 0)
                    update_data["data"]["summary"] = f"Writer generated draft (Revision {rev_count})"
                    update_data["data"]["draft_report_snippet"] = node_data["draft_report"][:200] + "..."
                elif node_name == "reviewer":
                    feedback = node_data.get("review_feedback")
                    if feedback and feedback.strip().upper() == "APPROVE":
                        update_data["data"]["summary"] = "Reviewer approved the report."
                    elif feedback:
                        update_data["data"]["summary"] = f"Reviewer requested revisions: {feedback[:100]}..."
                    else: # This case happens when END is forced by max revisions
                         update_data["data"]["summary"] = "Max revisions reached or review finished."
                         
                # Check if we reached the final report in this event
                if node_data.get("final_report"):
                    final_report = node_data["final_report"]
                elif node_data.get("draft_report") and node_data.get("next_agent") == "END": # Check if review ended
                    final_report = node_data["draft_report"] # Use draft if review ended here
                
                # Yield the event in SSE format
                yield f"data: {json.dumps(update_data)}\n\n"
                await asyncio.sleep(0.1) # Small delay
                current_step += 1
            else:
                logger.debug(f"Received non-dict event: {type(event)}")
                
        # After the stream finishes, determine final outcome
        if final_report:
            final_event = {"type": "final", "report": final_report}
            yield f"data: {json.dumps(final_event)}\n\n"
        else:
             # If no final report was explicitly set, try to get it from the last known state
            # This is a fallback - ideally final_report should be set explicitly
            logger.warning("Stream finished without explicit final report. Attempting fallback.")
            # Re-invoke to get the absolute final state (less efficient but robust)
            try:
                final_state_check = await app.ainvoke(initial_state)
                if final_state_check and final_state_check.get("final_report"):
                     final_event = {"type": "final", "report": final_state_check["final_report"]}
                     yield f"data: {json.dumps(final_event)}\n\n"
                elif final_state_check and final_state_check.get("draft_report"):
                     final_event = {"type": "final", "report": final_state_check["draft_report"]}
                     yield f"data: {json.dumps(final_event)}\n\n"
                else:
                     error_message = "Failed to retrieve final or draft report after execution."
            except Exception as invoke_err:
                 error_message = f"Error during final state check: {str(invoke_err)}"
                 
        if error_message:
            error_event = {"type": "error", "message": error_message}
            yield f"data: {json.dumps(error_event)}\n\n"

    except Exception as e:
        logger.exception(f"Error during graph execution stream for thread {thread_id}")
        error_message = f"Error generating report: {str(e)}"
        error_event = {"type": "error", "message": error_message}
        yield f"data: {json.dumps(error_event)}\n\n"
    finally:
        logger.info(f"--- Finished streaming for thread {thread_id} ---")

# --- API Endpoints --- #
@fastapi_app.post("/generate-report") # Remove response_model
async def generate_report_endpoint(request: GenerateRequest):
    """Receives a prompt and streams the LangGraph agent flow execution."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    thread_id = str(uuid.uuid4())
    logger.info(f"Received request for prompt: '{request.prompt[:50]}...' (Thread ID: {thread_id})")

    # Return a StreamingResponse that uses the async generator
    return StreamingResponse(
        stream_graph_events(request.prompt, thread_id),
        media_type="text/event-stream"
    )

# --- Root Endpoint (Optional) --- #
@fastapi_app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="<html><body><h1>LangGraph Report API</h1><p>Use POST /generate-report</p></body></html>")

# --- Health Check --- #
@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy"}

# --- How to Run (Instructions) --- #
# 1. Make sure you have .env file with OPENAI_API_KEY and TAVILY_API_KEY
# 2. Run from the project root directory:
#    uvicorn backend.main:fastapi_app --reload --port 8000 