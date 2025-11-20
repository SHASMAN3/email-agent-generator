# src/agent/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any
from langchain_core.messages.utils import get_buffer_string
from src.agent.state import AgentState
from src.agent.tools import tools, send_email  # We keep tools/send_email imports for graph compilation
# from src.core.llm import llm_with_tools # <--- REMOVED

# --- Dynamic Configuration Bridge ---
# This dictionary will be populated by app.py with the user's LLM and Tools
AGENT_CONFIG = {"llm_with_tools": None, "tools": None}

# --- Utility for Logging ---
def log_step(name: str, status: str, details: str = "") -> Dict[str, Any]:
    return {"node": name, "status": status, "details": details}

# --- Nodes (Functions) ---

def generate_draft(state: AgentState):
    """Generates the initial subject and body content based on the goal."""
    
    # CRITICAL: Fetch the dynamic LLM
    llm_with_tools = AGENT_CONFIG.get("llm_with_tools")
    if not llm_with_tools: raise Exception("LLM not configured in AGENT_CONFIG. Check app.py setup.")

    system_prompt = (
        "You are a professional Email Drafting Agent. Write a concise, professional "
        "email subject and body based on the user's goal. "
        "Your entire response MUST start with 'SUBJECT:' followed by the subject, and then a newline, "
        "followed by 'BODY:' followed by the full email body. Do not include any other text."
    )
    prompt = f"{system_prompt}\n\nGOAL: {state['goal']}\nRECIPIENT: {state['recipient']}"
    
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    
    # Universal Fix for Content Extraction
    try:
        raw_content = get_buffer_string([response])
    except:
        raw_content = str(response)

    try:
        # Strict parsing based on the prompt's required format
        subject_part, body_part = raw_content.split('\nBODY:', 1)
        subject = subject_part.replace('SUBJECT:', '').strip()
        body = body_part.strip()
        
        log = log_step("Draft_Creator", "Success", f"Subject drafted: {subject[:40]}...")
        
    except ValueError:
        subject = "ERROR: Drafting Failed"
        body = raw_content
        log = log_step("Draft_Creator", "Error", "LLM response did not follow strict SUBJECT:/BODY: format.")

    # Ensure logs are appended correctly
    current_logs = state.get("logs", [])
    current_logs.append(log)

    return {
        "subject": subject,
        "body": body,
        "logs": current_logs, 
        "messages": [AIMessage(content=f"Subject: {subject}\nBody: {body}")] 
    }

def review_and_decide(state: AgentState):
    """LLM node to review the draft and decide whether to send or provide feedback."""
    
    # CRITICAL: Fetch the dynamic LLM
    llm_with_tools = AGENT_CONFIG.get("llm_with_tools")
    if not llm_with_tools: raise Exception("LLM not configured in AGENT_CONFIG. Check app.py setup.")
    
    # CRITICAL PROMPT UPDATE FOR SENDING 
    system_prompt = (
        "You are the Email Orchestrator. Review the draft for professionalism and completeness. "
        "You have only two possible actions. The primary goal is to send a good email."
        "\n\nACTION 1: **SEND EMAIL**"
        "\nIf the draft is professional, complete, and ready to go, you **MUST** proceed directly to call the `send_email` tool, "
        "providing the exact `recipient`, `subject`, and `body`."
        "\n\nACTION 2: **REQUEST REVISION**"
        "\nIf the draft is lacking, unprofessional, or incomplete, you **MUST** respond ONLY with the word 'REVISION:', followed by your concise reason. DO NOT call the tool."
        "\n\nDECISION PRIORITY: Use ACTION 1 if the email is acceptable. Only use ACTION 2 if it's genuinely bad."
    )
    
    messages = [
        HumanMessage(content=system_prompt),
        AIMessage(content=f"Draft Subject: {state['subject']}\n\nDraft Body:\n---\n{state['body']}\n---")
    ]
    
    response = llm_with_tools.invoke(messages)
    
    current_logs = state.get("logs", [])
    log = log_step("Decision_Maker", "Complete", f"Decided to: {'Call Tool' if response.tool_calls else 'Give Feedback'}")
    current_logs.append(log)
    
    return {
        "review_feedback": response.content, 
        "messages": [response],
        "logs": current_logs
    }

def tool_executor(state: AgentState):
    """
    Executes the send_email tool, handling both explicit tool calls and forced execution 
    by synthesizing the arguments from the state.
    """
    # CRITICAL: Fetch the dynamic Tools list
    dynamic_tools = AGENT_CONFIG.get("tools")
    if not dynamic_tools: raise Exception("Tools not configured in AGENT_CONFIG. Check app.py setup.")

    tool_calls = state['messages'][-1].tool_calls if state['messages'] else None
    current_logs = state.get("logs", [])
    
    # --- Check 1: Explicit Tool Call Found ---
    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Use the dynamic tools list
        tool_to_run = next(t for t in dynamic_tools if t.name == tool_name)
        tool_result = tool_to_run.invoke(tool_args)
        
    # --- Check 2: No Explicit Tool Call (Forced Execution) ---
    else:
        # Synthesize arguments directly from the state for the forced execution path.
        try:
            # Find the specific send_email tool object from the dynamic list
            send_email_tool = next(t for t in dynamic_tools if t.name == "send_email")
            
            # FINAL FIX: Call the tool using .invoke() and pass arguments as a dict 
            tool_result = send_email_tool.invoke({
                "recipient": state['recipient'],
                "subject": state['subject'],
                "body": state['body']
            })
            
            log = log_step("Tool_Executor", "Forced_Success", "Tool call was synthesized and executed.")
            current_logs.append(log)
            # Return result and exit node
            return {"status": tool_result, "logs": current_logs, "messages": []}

        except Exception as e:
            # Handle potential exceptions during manual call 
            log = log_step("Tool_Executor", "Forced_Error", f"Manual tool execution failed. Check .env: {str(e)}")
            current_logs.append(log)
            # Return error result and exit node
            return {"status": f"ERROR: Manual execution failed. Details: {str(e)}", "logs": current_logs, "messages": []}


    log = log_step("Tool_Executor", "Complete", f"Tool output: {tool_result}")
    current_logs.append(log)
    
    return {"status": tool_result, "logs": current_logs, "messages": []}

def route_next_step(state: AgentState):
    """
    Conditional edge to decide where to go after review_and_decide. 
    Implements a fail-safe: if no tool call but no explicit rejection, force tool execution.
    """
    # Check 1: Tool call is explicitly made
    if state['messages'][-1].tool_calls:
        return "tool_executor"
    
    # Check 2: Explicit rejection text is found (using the REVISION keyword)
    elif "REVISION:" in state['review_feedback']:
         return "END_FEEDBACK"

    # Fail-Safe: If no explicit tool call or rejection was made, we force tool execution.
    else:
        return "tool_executor" 

# --- Build the Graph ---

def build_email_agent():
    """
    Compiles and returns the LangGraph agent. 
    It relies on AGENT_CONFIG being set by app.py before invocation.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("draft_creator", generate_draft)
    workflow.add_node("decision_maker", review_and_decide)
    workflow.add_node("tool_executor", tool_executor)

    workflow.set_entry_point("draft_creator")

    workflow.add_edge("draft_creator", "decision_maker")
    
    workflow.add_conditional_edges(
        "decision_maker",
        route_next_step,
        {
            "tool_executor": "tool_executor", 
            "END_FEEDBACK": END              
        }
    )
    
    workflow.add_edge("tool_executor", END)
    
    return workflow.compile()

# Initialize the agent (Compiler will use the imports from the top of the file)
email_agent = build_email_agent()