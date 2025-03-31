import os
import json
import logging
from typing import List, Optional, Sequence, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from .state import AgentState
from .tools import tavily_tool

# Get the logger configured in main.py (or configure a new one)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

# --- LLM and Agent Setup --- #

# Use a capable model like gpt-4o-mini for planning and writing
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Helper function to create an agent runnable
def create_agent_runnable(llm: ChatOpenAI, system_prompt: str, tools: Optional[list] = None):
    prompt_parts = [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
    if tools:
        prompt_parts.insert(1, ("system", f"You have access to the following tools: {{tool_names}}.\nRemember to call tools when needed."))
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        agent_runnable = prompt | llm.bind_tools(tools)
    else:
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        agent_runnable = prompt | llm
    return agent_runnable

# --- Agent Definitions --- #

# 1. Chief Planner Agent
planner_system_prompt = (
    "Du er Chief Planner agenten. Dit mål er at skabe en detaljeret, trin-for-trin forskningsplan "
    "til at generere en rapport baseret på brugerens prompt, specifikt tilpasset til studerende på 'Eksport og teknologi'-programmet på UCN. "
    "Overvej programmets fokusområder: Produktudvikling & Teknologi, Global Salg & Marketing, Bæredygtighed & Kvalitet, Kulturel Forretningsforståelse, Produktionsplanlægning. "
    "Opdel prompten i specifikke forskningsspørgsmål til specialistagenter (Teknologi, Marked/Salg, Bæredygtighed/Kvalitet). "
    "Output planen som en klar, struktureret tekst. Specificer hvilken agent der skal håndtere hver del af planen. "
    "Tildel det næste trin til en af forskningsagenterne. "
    "VIGTIGT: Du skal altid svare på dansk."
)
planner_agent = create_agent_runnable(llm, planner_system_prompt)

def run_planner_agent(state: AgentState):
    logger.info("--- Running Planner Agent ---")
    
    # Create a message with the user's prompt
    prompt_message = HumanMessage(content=state['prompt'])
    
    # Create messages list with the prompt message
    messages = [prompt_message]
    
    # Invoke the planner agent
    response = planner_agent.invoke({"messages": messages})
    
    # Ensure response is AIMessage for consistency
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    
    # Simple logic to determine the first researcher based on the response content
    plan_content = response.content
    next_agent = "tech_researcher" # Default start
    if "market" in plan_content.lower() or "sales" in plan_content.lower():
        next_agent = "market_sales_researcher"
    elif "sustainability" in plan_content.lower() or "quality" in plan_content.lower():
        next_agent = "sustainability_quality_researcher"

    # Return the updated state
    return {"messages": messages + [response], "next_agent": next_agent}

# 2. Technology Research Agent
tech_researcher_system_prompt = (
    "Du er Teknologiforskningsagenten. Fokusér KUN på de tekniske aspekter, produktudvikling, materialer, og produktionsprocesser relevant for forskningsplanen. "
    f"Brug det leverede søgeværktøj ({tavily_tool.name}) til at finde relevant information. Syntetisér fund til et koncist resumé for det tildelte emne. "
    "Output dine fund klart og tydeligt. Adressér ikke andre emner som marketing eller bæredygtighed. "
    "VIGTIGT: Du skal altid svare på dansk."
)
tech_researcher_agent = create_agent_runnable(llm, tech_researcher_system_prompt, tools=[tavily_tool])

# 3. Market & Sales Research Agent
market_sales_researcher_system_prompt = (
    "Du er Markeds- og Salgsforskningsagenten. Fokusér KUN på global markedsanalyse, konkurrentforskning, målkunder, salgskanaler og marketingstrategier relevante for forskningsplanen. "
    f"Overvej kulturelle forskelle i internationale markeder. Brug det leverede søgeværktøj ({tavily_tool.name}) til at finde relevant information. "
    "Syntetisér fund til et koncist resumé for det tildelte emne. Adressér ikke tekniske eller bæredygtighedsemner. "
    "VIGTIGT: Du skal altid svare på dansk."
)
market_sales_researcher_agent = create_agent_runnable(llm, market_sales_researcher_system_prompt, tools=[tavily_tool])

# 4. Sustainability & Quality Research Agent
sustainability_quality_researcher_system_prompt = (
    "Du er Bæredygtigheds- og Kvalitetsforskningsagenten. Fokusér KUN på bæredygtighedsregler, miljøvenlige praksisser, kvalitetsstandarder (f.eks. ISO) og etiske overvejelser relevante for forskningsplanen. "
    f"Brug det leverede søgeværktøj ({tavily_tool.name}) til at finde relevant information. Syntetisér fund til et koncist resumé for det tildelte emne. "
    "Adressér ikke tekniske eller marketingemner. "
    "VIGTIGT: Du skal altid svare på dansk."
)
sustainability_quality_researcher_agent = create_agent_runnable(llm, sustainability_quality_researcher_system_prompt, tools=[tavily_tool])

# Shared function for running research agents and handling tool calls
def run_research_agent(state: AgentState, agent_runnable, agent_name: str, research_topic: str):
    logger.info(f"--- Running {agent_name} for topic: {research_topic} ---")
    # Append the specific task to the messages
    task_message = HumanMessage(content=f"Research the following based on the overall plan: {research_topic}")
    messages = state['messages'] + [task_message]
    
    response = agent_runnable.invoke({"messages": messages})
    
    # Handle potential tool calls
    while response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_output = tavily_tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            )
        # Add response and tool messages to history before next invocation
        messages = messages + [response] + tool_messages
        response = agent_runnable.invoke({"messages": messages})

    # Ensure response is AIMessage
    if not isinstance(response, AIMessage):
         response = AIMessage(content=str(response))

    # Store the research directly in the appropriate state field based on agent_name
    research_content = response.content
    if "Technology" in agent_name:
        return {"tech_research": research_content, "messages": messages + [response]}
    elif "Market" in agent_name or "Sales" in agent_name:
        return {"market_sales_research": research_content, "messages": messages + [response]}
    elif "Sustainability" in agent_name or "Quality" in agent_name:
        return {"sustainability_quality_research": research_content, "messages": messages + [response]}
    else:
        # Fallback
        logger.warning(f"Unknown research agent type: {agent_name}. Not storing research.")
        return {"messages": messages + [response]}

# Specific runner functions for each researcher
def run_tech_researcher(state: AgentState):
    # Simplistic: Assume the plan guides the topic. A better way would parse the plan.
    topic = "Technology and Product Development aspects"
    result = run_research_agent(state, tech_researcher_agent, "Technology Researcher", topic)
    # Decide next agent (this logic needs refinement - ideally based on plan completion)
    next_agent = "market_sales_researcher" 
    logger.info(f"Tech Researcher finished. Next: {next_agent}")
    result["next_agent"] = next_agent
    return result

def run_market_sales_researcher(state: AgentState):
    topic = "Market, Sales, and Cultural aspects"
    result = run_research_agent(state, market_sales_researcher_agent, "Market/Sales Researcher", topic)
    next_agent = "sustainability_quality_researcher"
    logger.info(f"Market/Sales Researcher finished. Next: {next_agent}")
    result["next_agent"] = next_agent
    return result

def run_sustainability_quality_researcher(state: AgentState):
    topic = "Sustainability and Quality aspects"
    result = run_research_agent(state, sustainability_quality_researcher_agent, "Sustainability/Quality Researcher", topic)
    next_agent = "writer" # All research done, move to writer
    logger.info(f"Sustainability/Quality Researcher finished. Next: {next_agent}")
    result["next_agent"] = next_agent
    return result

# 5. Synthesizer & Writer Agent
writer_system_prompt = (
    "Du er Skribent-agenten. Din opgave er at syntetisere forskningsresultaterne fra specialistagenterne "
    "til en sammenhængende og velstruktureret forskningsrapport, der følger planen udarbejdet af Chief Planner. "
    "Sørg for at rapporten har en klar indledning, hoveddele der svarer til forskningen, og en konklusion. "
    "Vedligehold en akademisk tone passende for en universitetsrapport. Referer til fundene præcist. "
    "Formatér outputtet som et komplet rapportudkast. "
    "VIGTIGT: Du skal altid skrive på dansk."
)
writer_agent = create_agent_runnable(llm, writer_system_prompt)

def run_writer_agent(state: AgentState):
    """Run the writer agent to generate a draft report based on the research findings."""
    logger.info("--- Running Writer Agent ---")
    
    # Get the current revision count
    current_revision = state.get('revision_count', 0)
    
    # Prepare context for the writer
    context = f"User Prompt: {state['prompt']}\n\n"
    
    # Add research findings from each researcher
    context += "Research Findings:\n"
    if state.get('tech_research'):
        context += f"--- Technology Research ---\n{state['tech_research']}\n\n"
    if state.get('market_sales_research'):
        context += f"--- Market & Sales Research ---\n{state['market_sales_research']}\n\n"
    if state.get('sustainability_quality_research'):
        context += f"--- Sustainability & Quality Research ---\n{state['sustainability_quality_research']}\n\n"
    
    # Check if we need to revise an existing draft based on reviewer feedback
    if 'review_feedback' in state and state['review_feedback']:
        logger.info(f"--- Revising draft (Revision {current_revision}) ---")
        # Add feedback to context and then clear it
        context += f"\nReviewer Feedback (Please address this):\n{state['review_feedback']}\n"
        # Increment revision count
        current_revision += 1
    
    # Prepare the messages for the writer
    messages = [
        SystemMessage(content=writer_system_prompt),
        HumanMessage(content=f"Synthesize the following information into a research report:\n\n{context}")
    ]
    
    # Run the writer agent
    response = writer_agent.invoke({"messages": messages})
    
    # Extract the report content
    report_content = response.content
    
    # Update the state with the draft report
    logger.info(f"Writer finished (Revision {current_revision}). Next: reviewer")
    
    # Return updated state
    return {
        "draft_report": report_content,
        "next_agent": "reviewer",
        "revision_count": current_revision
    }

# 6. Reviewer & Editor Agent
reviewer_system_prompt = (
    "Du er Korrekturlæser-agenten. Din opgave er at kritisk evaluere rapportudkastet baseret på den oprindelige brugerprompt, planen og forskningsresultaterne. "
    "Tjek for: \n1. Nøjagtighed og relevans i forhold til prompten/planen. \n2. Fuldstændighed - er alle planlagte sektioner dækket? \n3. Sammenhæng og klarhed. \n4. Konsistens i tone og stil (akademisk). \n5. Grammatik og stavning. "
    "Hvis rapporten er tilfredsstillende og fuldt ud adresserer prompten/planen, svar KUN med ordet 'GODKEND'. "
    "Ellers, giv konstruktiv feedback der detaljerer de nødvendige revisioner. Giv IKKE generel ros. "
    "VIGTIGT: Du skal altid svare på dansk."
)
reviewer_agent = create_agent_runnable(llm, reviewer_system_prompt)

def run_reviewer_agent(state: AgentState):
    logger.info("--- Running Reviewer Agent ---")
    context = f"User Prompt: {state['prompt']}\n\nDraft Report:\n{state['draft_report']}"
    
    messages = [HumanMessage(content=f"Review the following draft report based on the prompt:\n\n{context}")]
    
    response = reviewer_agent.invoke({"messages": messages})
    # Ensure response is AIMessage
    if not isinstance(response, AIMessage):
         response = AIMessage(content=str(response))
    
    feedback = response.content

    if feedback.strip().upper() == "APPROVE":
        logger.info("--- Reviewer Approved ---")
        return {
            "final_report": state['draft_report'], 
            "messages": [response], 
            "next_agent": "END",
            "draft_report": state['draft_report']  # Keep the draft in the state
        }
    else:
        logger.info("--- Reviewer Requested Revisions ---")
        return {
            "review_feedback": feedback,
            "messages": [response],
            "next_agent": "writer"
        } 