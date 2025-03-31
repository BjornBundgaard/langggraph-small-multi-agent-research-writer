import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_tavily_tool():
    """Initializes and returns the Tavily search tool."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables. Please add it to your .env file.")
    
    # You can adjust max_results if needed
    search_tool = TavilySearchResults(max_results=5, api_key=tavily_api_key)
    search_tool.description = "A search engine optimized for comprehensive, accurate, and trusted results. Useful for researching complex topics, facts, and finding specific information online."
    return search_tool

# Instantiate the tool for easy import
tavily_tool = get_tavily_tool() 