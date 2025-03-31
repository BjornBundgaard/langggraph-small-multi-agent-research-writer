import logging # Import logging
from langgraph.graph import StateGraph, END, START
from backend.state import AgentState
from backend.agents import (
    run_planner_agent,
    run_tech_researcher,
    run_market_sales_researcher,
    run_sustainability_quality_researcher,
    run_writer_agent,
    run_reviewer_agent
)
from langchain_core.tracers.langchain import wait_for_all_tracers

# Get the logger configured in main.py
logger = logging.getLogger(__name__)

# Define the nodes for the graph
nodes = {
    "planner": run_planner_agent,
    "tech_researcher": run_tech_researcher,
    "market_sales_researcher": run_market_sales_researcher,
    "sustainability_quality_researcher": run_sustainability_quality_researcher,
    "writer": run_writer_agent,
    "reviewer": run_reviewer_agent,
}

# Define the conditional routing logic
def route_after_planner(state: AgentState):
    # Based on the planner's decision (stored in next_agent)
    logger.info(f"--- Routing after Planner: Next agent is {state['next_agent']} ---")
    return state['next_agent']

def route_after_research(state: AgentState):
    next_agent = state.get('next_agent') # Use .get for safety
    logger.info(f"--- Routing after Research: Next agent is {next_agent} ---")
    if next_agent == "writer":
        return "writer"
    elif next_agent == "market_sales_researcher":
        return "market_sales_researcher"
    elif next_agent == "sustainability_quality_researcher":
        return "sustainability_quality_researcher"
    else:
        # Fallback or error case
        logger.warning(f"Unexpected next_agent '{next_agent}' after research. Defaulting to writer.")
        return "writer"

MAX_REVISIONS = 1

def route_after_review(state: AgentState):
    next_agent = state.get('next_agent') 
    revision_count = state.get('revision_count', 0)
    logger.info(f"--- Routing after Review (Revision count: {revision_count}): Next agent is {next_agent} ---")
    
    if next_agent == "writer" and revision_count >= MAX_REVISIONS:
        logger.warning(f"Max revisions ({MAX_REVISIONS}) reached. Forcing END.")
        
        # Get the draft report, either from the state or from a node
        draft_report = None
        
        # First try to get it from the top-level state
        if "draft_report" in state and state["draft_report"]:
            draft_report = state["draft_report"]
            logger.info(f"Using draft_report from top-level state")
            
        # If not found, check if it's in the 'writer' node
        elif "writer" in state and isinstance(state["writer"], dict) and "draft_report" in state["writer"]:
            draft_report = state["writer"]["draft_report"]
            logger.info(f"Using draft_report from writer node")
            
        # If we found a draft report, set it as the final report
        if draft_report:
            # Make sure the final_report is set at the top level of the state
            # This is the key fix - ensuring the final_report is directly in the state dict
            state["final_report"] = draft_report
            logger.info(f"Final report set from draft: {state['final_report'][:50]}...")
            
            # Debug: Log the full state structure
            logger.info(f"Final state structure: {list(state.keys())}")
        else:
            logger.error("No draft report available to set as final report")
            logger.error(f"Current state keys: {list(state.keys())}")
            
        return END
    elif next_agent == "writer":
        return "writer"
    elif next_agent == "END":
        # This happens if the reviewer approved
        return END
    else:
        # Fallback or error case
        logger.warning(f"Unexpected next_agent '{next_agent}' after review. Defaulting to END.")
        return END

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", run_planner_agent)
workflow.add_node("tech_researcher", run_tech_researcher)
workflow.add_node("market_sales_researcher", run_market_sales_researcher)
workflow.add_node("sustainability_quality_researcher", run_sustainability_quality_researcher)
workflow.add_node("writer", run_writer_agent)
workflow.add_node("reviewer", run_reviewer_agent)

# Define edges and conditional routing
workflow.add_edge(START, "planner")

# Planner decides the first researcher
workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "tech_researcher": "tech_researcher",
        "market_sales_researcher": "market_sales_researcher",
        "sustainability_quality_researcher": "sustainability_quality_researcher",
    }
)

# Research agents route sequentially or to the writer
workflow.add_conditional_edges(
    "tech_researcher",
    route_after_research,
    {
        "market_sales_researcher": "market_sales_researcher",
        "sustainability_quality_researcher": "sustainability_quality_researcher",
        "writer": "writer"
    }
)
workflow.add_conditional_edges(
    "market_sales_researcher",
    route_after_research,
    {
        "sustainability_quality_researcher": "sustainability_quality_researcher",
        "writer": "writer"
    }
)
workflow.add_conditional_edges(
    "sustainability_quality_researcher",
    route_after_research,
    {
        "writer": "writer"
    }
)

# Writer always goes to reviewer
workflow.add_edge("writer", "reviewer")

# Reviewer routes back to writer or ends the process
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "writer": "writer",
        END: END
    }
)

# Compile the graph with config
app = workflow.compile(
    checkpointer=None,  # Disable checkpointing for now
    interrupt_before=["reviewer"],  # Allow interruption before review for human-in-the-loop
    interrupt_after=["writer"],    # Allow interruption after writing for human-in-the-loop
)

logger.info("LangGraph compiled successfully!")

# This ensures all traces are properly captured
def cleanup_traces():
    wait_for_all_tracers()

# Register cleanup function if needed
import atexit
atexit.register(cleanup_traces) 