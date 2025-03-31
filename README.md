# LangGraph Multiagent System

## Overview
This project is an implementation of a multi-agent system built with LangGraph and LangChain. The system consists of several specialized agents that collaborate to generate comprehensive research reports based on user input.

## System Architecture
The system consists of the following agents working together in an orchestrated workflow:

1. **Chief Planner Agent**: Creates a detailed research plan based on the user's prompt.
2. **Technology Research Agent**: Collects technological information related to the topic.
3. **Market & Sales Research Agent**: Conducts market analysis and investigates sales strategies.
4. **Sustainability & Quality Research Agent**: Focuses on sustainability and quality standards.
5. **Writer Agent**: Synthesizes research findings into a coherent report.
6. **Reviewer Agent**: Evaluates and provides feedback on the report.

## Installation

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installing Dependencies
Install the required packages with the following command:

```bash
pip install -r requirements.txt

**Create a .env file in the project's root directory and add the following API keys:**
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=your_project_name



Running the Application
Backend
Start the backend server with:
uvicorn backend.main:fastapi_app --reload --port 8000

Frontend
In a separate terminal, start the frontend Streamlit application with:
streamlit run frontend/app.py

Using the Application
Open your browser and go to http://localhost:8501

Enter your topic or question in the text field

Click the "Generate Report" button

View real-time updates as the agents work using Server-Sent Events

The final result will be displayed in the user interface when completed

LangGraph Server (Optional)
To use LangGraph Studio for visualization and debugging:

Install LangGraph CLI:

bash
Copy
Edit
pip install langgraph-cli
Start the LangGraph server:

bash
Copy
Edit
langgraph dev
Open LangGraph Studio in your browser at http://127.0.0.1:2024

Features
Multilingual Support: All agents communicate in Danish

Real-time Updates: Watch agents work in real-time via Server-Sent Events

Automatic Error Handling: Limits retries and prevents infinite loops

Robust Report Generation: Produces academic reports with citations and structured formatting

Troubleshooting
If you encounter issues connecting to the LangGraph server, ensure environment variables are properly configured

Check that all required API keys are valid and have sufficient permissions

For backend server issues, check log files for specific error messages

Contributing
Contributions to this project are welcome. Follow these steps to contribute:

Fork the project

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

License
This project is licensed under the MIT License. See the LICENSE file for details.

Copy
Edit
