# app.py
import streamlit as st
import os
from langchain_core.tools import StructuredTool # For dynamic tool creation
from langchain_core.utils.function_calling import convert_to_openai_function
from src.agent.state import AgentState
# Import the Pydantic schema and the raw sending function
from src.agent.tools import SendEmailArgsSchema, send_email_func 
from src.core.llm import create_llm # Import the LLM factory function
from src.agent.graph import build_email_agent, AGENT_CONFIG # Import the graph constructor and the config bridge

# --- Configuration & Setup ---

st.set_page_config(page_title="Gemini & LangGraph Email Agent", layout="wide")
st.title("📧 AI-Powered Email Agent")
st.markdown("---")

# Use constants for host/port
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587

# --- Collect Credentials in Streamlit UI ---
st.sidebar.header("🔑 User Credentials")
gemini_key = st.sidebar.text_input("GEMINI API Key:", type="password", help="Your personal key for the Gemini model.")
smtp_username = st.sidebar.text_input("SMTP Username (Sender Email):", type="password", help="Your sending Gmail address.")
smtp_password = st.sidebar.text_input("SMTP App Password:", type="password", help="Your 16-character Google App Password (required if 2FA is on).")
st.sidebar.caption("All credentials are used only for this session.")

# --- Session State and Input Fields ---

if 'goal' not in st.session_state: st.session_state.goal = ""
if 'recipient' not in st.session_state: st.session_state.recipient = ""
if 'email_state' not in st.session_state: st.session_state.email_state = None

col_input, col_recipient = st.columns([3, 1])
with col_recipient:
    recipient_input = st.text_input("Recipient Email:", value=st.session_state.recipient, key="recipient_input", placeholder="name@example.com")
with col_input:
    goal_input = st.text_area("Email Goal (What should the email achieve?):", value=st.session_state.goal, key="goal_input", height=100, placeholder="e.g., Ask Sarah for the updated Q4 sales figures and schedule a quick sync.")


# --- Agent Execution Function ---

def run_agent():
    st.session_state.goal = st.session_state.goal_input
    st.session_state.recipient = st.session_state.recipient_input
    
    # 1. Input Validation
    if not st.session_state.goal or not st.session_state.recipient:
        st.error("Please enter both the Email Goal and the Recipient.")
        return
    if not gemini_key or not smtp_password or not smtp_username:
        st.error("Please enter all required API keys and SMTP credentials in the sidebar.")
        return

    with st.spinner(f"Running LangGraph agent for goal: **{st.session_state.goal}**..."):
        
        # 2. Dynamic Tool Creation (using user-provided credentials)
        try:
            # Create a tool instance that closes over the user's credentials
            dynamic_send_email_tool = StructuredTool.from_function(
                func=lambda recipient, subject, body: send_email_func(
                    recipient=recipient, 
                    subject=subject, 
                    body=body,
                    host=EMAIL_HOST,
                    port=EMAIL_PORT,
                    username=smtp_username,
                    password=smtp_password
                ),
                name="send_email",
                description="A tool to send a completed email to a specified recipient.",
                # 🛑 FIX: Use the imported Pydantic class as the args_schema
                args_schema=SendEmailArgsSchema 
            )
            dynamic_tools = [dynamic_send_email_tool]
            
        except Exception as e:
            st.error(f"Error creating tool: {e}. Check function signature in tools.py.")
            return

        # 3. Dynamic LLM Initialization
        llm = create_llm(api_key=gemini_key)
        
        # 4. Bind LLM to Tools
        llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in dynamic_tools])
        
        # 5. Set the global AGENT_CONFIG for graph.py to use
        AGENT_CONFIG["llm_with_tools"] = llm_with_tools
        AGENT_CONFIG["tools"] = dynamic_tools
        
        # 6. Re-compile and Run the Agent
        # Re-compile to ensure the LangGraph nodes use the newly configured AGENT_CONFIG
        email_agent = build_email_agent() 
        
        initial_state = AgentState(
            goal=st.session_state.goal,
            recipient=st.session_state.recipient,
            subject="",
            body="",
            review_feedback="",
            status="",
            logs=[],
            messages=[]
        )
        
        try:
            # Run the entire graph. Call invoke on the compiled agent object.
            final_state = email_agent.invoke(initial_state)
            st.session_state.email_state = final_state
            st.success("Agent workflow completed!")
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            st.session_state.email_state = None

# --- Main App Body ---

# Button to start the process
if st.button("Generate & Decide to Send", type="primary"):
    run_agent()

# --- Display Results and Logs ---
if st.session_state.email_state:
    state = st.session_state.email_state
    st.markdown("---")
    
    # 1. Display Execution Logs
    st.header("📑 Execution Logs")
    if state.get('logs'):
        log_data = [{'Node': log['node'], 'Status': log['status'], 'Details': log['details']} for log in state['logs']]
        st.table(log_data)
    else:
        st.info("No explicit execution logs found in the final state.")

    st.markdown("---")

    # 2. Display Final Result
    
    # Check 1: Success/Failure from the Tool Executor node (Status is present)
    if state['status']:
        
        if "SUCCESS" in state['status']:
            st.header("✅ Email Sent!")
            st.success(f"Final Status: {state['status']}")
        else:
            st.header("❌ Tool Execution Error")
            # If status is an error, it's now a credential error, not a tool error
            st.error(f"Final Status: {state['status']}. Check your SMTP credentials in the sidebar.")
            
        st.subheader(f"Generated Subject: {state['subject']}")
        st.code(state['body'], language="markdown")

    # Check 2: Feedback from the Decision Maker node (Status is EMPTY, meaning conditional END was met)
    elif "REVISION:" in state['review_feedback']:
        
        st.header("⚠️ Draft Requires Review")
        st.warning("The agent decided **not** to send the email.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Generated Subject: {state['subject']}")
            st.code(state['body'], language="markdown")
        
        with col2:
            st.subheader("Agent Feedback")
            feedback_text = state['review_feedback'].replace("REVISION:", "").strip()
            if feedback_text:
                 st.error(feedback_text)
            else:
                 st.info("The agent decided the draft was not ready but produced no explicit feedback. Provide a more detailed goal to proceed.")
        
    # Final Catch-all (for genuine unknown issues)
    else:
        st.header("❓ Unknown Final State")
        st.json(state)