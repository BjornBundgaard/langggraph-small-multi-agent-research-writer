import streamlit as st
import requests
import json
import httpx # Use httpx for streaming support in Streamlit

# --- Configuration --- #
BACKEND_URL = "http://localhost:8000/generate-report" # Updated URL

# --- Streamlit UI --- #
st.set_page_config(page_title="Multi-Agent Research Report Generator", layout="wide")
st.title("🕵️‍♂️ Multi-Agent Research Report Generator")

# Input prompt
prompt = st.text_area("Enter your research topic:", height=100)

# Generate button
if st.button("Generate Report"):
    if not prompt:
        st.warning("Please enter a research topic.")
    else:
        st.info("Generating report... Please wait.")
        
        # Placeholders for updates and final report
        status_placeholder = st.empty()
        report_placeholder = st.empty()
        
        status_updates = []
        final_report_content = None
        error_content = None

        try:
            # Use httpx.stream for Server-Sent Events
            with httpx.stream("POST", BACKEND_URL, json={"prompt": prompt}, timeout=None) as response:
                if response.status_code == 200:
                    # Process the event stream
                    for line in response.iter_lines():
                        if line.startswith("data:"):
                            try:
                                data = json.loads(line[len("data:"):])
                                
                                if data.get("type") == "update":
                                    step = data.get("step", "?")
                                    node = data.get("node", "Unknown")
                                    summary = data.get("data", {}).get("summary", "Processing...")
                                    update_text = f"**Step {step} ({node}):** {summary}"
                                    status_updates.append(update_text)
                                    # Display latest status updates
                                    status_placeholder.markdown("\n".join(status_updates))
                                    
                                    # Optionally display draft snippet
                                    # snippet = data.get("data", {}).get("draft_report_snippet")
                                    # if snippet:
                                    #     report_placeholder.info(f"Current Draft Snippet:\n```markdown\n{snippet}\n```")
                                        
                                elif data.get("type") == "final":
                                    final_report_content = data.get("report")
                                    break # Exit loop once final report is received
                                    
                                elif data.get("type") == "error":
                                    error_content = data.get("message", "An unknown error occurred.")
                                    break # Exit loop on error
                                    
                            except json.JSONDecodeError:
                                st.warning(f"Received non-JSON data: {line}")
                            except Exception as e:
                                st.error(f"Error processing stream data: {e}")
                                error_content = f"Frontend error processing stream: {e}"
                                break
                else:
                     error_content = f"Backend error: {response.status_code} - {response.text}"

        except httpx.RequestError as e:
            error_content = f"Failed to connect to the backend API: {e}"
        except Exception as e:
            error_content = f"An unexpected error occurred: {e}"

        # --- Display Final Results --- #
        if final_report_content:
            status_placeholder.success("Report generation complete!")
            report_placeholder.markdown("## Generated Report")
            report_placeholder.markdown(final_report_content)
        elif error_content:
            status_placeholder.error(f"Error generating report: {error_content}")
            report_placeholder.empty() # Clear any previous snippets
        else:
            # This case might happen if the stream ends without a final or error event
            status_placeholder.warning("Report generation finished, but no final report or error was received.")

# --- Instructions/Notes --- #
st.sidebar.title("How to Use")
st.sidebar.markdown("""
1.  **Enter Prompt:** Type the subject or research question for your report in the text area.
2.  **Generate:** Click the 'Generate Report' button.
3.  **Wait:** The AI agents will now collaborate. This involves planning, multiple web searches, writing, and reviewing, so it can take a few minutes.
4.  **View:** The final report draft will appear below.
""")
st.sidebar.title("Backend Status")
st.sidebar.markdown(f"API Endpoint: `{BACKEND_URL}`")
st.sidebar.markdown("Ensure the backend server is running (`uvicorn backend.main:fastapi_app --reload --port 8000`) in your terminal.") 