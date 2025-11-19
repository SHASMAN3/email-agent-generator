# app.py
import streamlit as st
from dotenv import load_dotenv
import os
from src.agent.graph import email_agent
from src.agent.state import AgentState

# Load environment variables
load_dotenv()
if not os.getenv("GEMINI_API_KEY") or not os.getenv("SMTP_APP_PASSWORD"):
    st.error("Missing GEMINI_API_KEY or SMTP_APP_PASSWORD in .env file. Please check setup.")
    st.stop()

st.set_page_config(page_title="Gemini & LangGraph Email Agent", layout="wide")
st.title("📧 AI-Powered Email Agent")

# --- Sidebar and Initialization (Unchanged) ---
with st.sidebar:
    st.header("Workflow")
    st.write(
        "1. **Input Goal & Recipient**.\n"
        "2. **Draft:** Gemini creates the Subject and Body (Node 1).\n"
        "3. **Review/Decide:** Gemini reviews and either:\n"
        "   - Calls the **`send_email` Tool** (Node 3) to dispatch the email and **ENDS**.\n"
        "   - Gives **FEEDBACK** (Node 2) and **ENDS** the workflow.\n"
    )
    st.markdown("---")
    st.caption("Built with LangGraph, Gemini 2.5 Flash, and Streamlit.")

if 'goal' not in st.session_state: st.session_state.goal = ""
if 'recipient' not in st.session_state: st.session_state.recipient = ""
if 'email_state' not in st.session_state: st.session_state.email_state = None

col_input, col_recipient = st.columns([3, 1])
with col_recipient:
    recipient_input = st.text_input("Recipient Email:", value=st.session_state.recipient, key="recipient_input", placeholder="name@example.com")
with col_input:
    goal_input = st.text_area("Email Goal (What should the email achieve?):", value=st.session_state.goal, key="goal_input", height=100, placeholder="e.g., Ask Sarah for the updated Q4 sales figures and schedule a quick sync.")

def run_agent():
    st.session_state.goal = st.session_state.goal_input
    st.session_state.recipient = st.session_state.recipient_input
    
    if not st.session_state.goal or not st.session_state.recipient:
        st.error("Please enter both the Email Goal and the Recipient.")
        return
    
    with st.spinner(f"Running LangGraph agent for goal: **{st.session_state.goal}**..."):
        initial_state = AgentState(
            goal=st.session_state.goal,
            recipient=st.session_state.recipient,
            subject="",
            body="",
            review_feedback="",
            status="",
            logs=[],  # Important: Initialize logs as an empty list
            messages=[]
        )
        
        try:
            # Run the entire graph. The .invoke() call now returns the final state.
            final_state = email_agent.invoke(initial_state)
            st.session_state.email_state = final_state
            st.success("Agent workflow completed!")
        except Exception as e:
            st.error(f"An error occurred during agent execution: {e}")
            st.session_state.email_state = None

# app.py (FINAL DISPLAY LOGIC SECTION)

# ... (Previous code)

# Button to start the process
if st.button("Generate & Decide to Send", type="primary"):
    run_agent()

# --- Display Results and Logs (UPDATED FOR ROBUST ENDING) ---
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
            # If status is an error, it's usually an authentication issue
            st.error(f"Final Status: {state['status']}. Check your SMTP credentials in .env file.")
            
        st.subheader(f"Generated Subject: {state['subject']}")
        st.code(state['body'], language="markdown")

    # Check 2: Feedback from the Decision Maker node (Status is EMPTY, meaning conditional END was met)
    elif "FEEDBACK" in state['review_feedback'] or (not state['status'] and state.get('review_feedback') is not None and len(state['logs']) > 1):
        
        st.header("⚠️ Draft Requires Review")
        st.warning("The agent decided **not** to send the email.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Generated Subject: {state['subject']}")
            st.code(state['body'], language="markdown")
        
        with col2:
            st.subheader("Agent Feedback")
            feedback_text = state['review_feedback'].replace("FEEDBACK:", "").strip()
            if feedback_text:
                 st.error(feedback_text)
            else:
                 st.info("The agent decided the draft was not ready but produced no explicit feedback. Provide a more detailed goal to proceed.")
        
    # Final Catch-all (for genuine unknown issues)
    else:
        st.header("❓ Unknown Final State")
        st.json(state)