from typing import Optional
from typing_extensions import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage
from typing import Sequence

# Define the structure for the overall state of the graph
class AgentState(TypedDict):
    # User's initial request
    prompt: str

    # Store research results from each researcher
    tech_research: Optional[str]
    market_sales_research: Optional[str]
    sustainability_quality_research: Optional[str]

    # Review feedback on drafts
    review_feedback: Optional[str]

    # The draft report for review
    draft_report: Optional[str]

    # The final report to return to the user
    final_report: Optional[str]

    # Keep track of the next agent to run
    next_agent: Optional[str]

    # Add a counter for revision loops
    revision_count: int

    # Messages history for context (especially for agent interactions)
    # Using Annotated sequence with operator.add allows messages to be appended
    messages: Annotated[Sequence[BaseMessage], operator.add] 