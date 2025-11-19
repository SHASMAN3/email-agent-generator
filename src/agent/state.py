# src/agent/state.py
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the LangGraph workflow for email sending.
    """
    goal: str
    recipient: str
    subject: str
    body: str
    review_feedback: str
    status: str
    
    # CORRECTED: Logs are NOT annotated with add_messages.
    logs: List[Dict[str, Any]] 
    
    # Messages use the annotation for incremental history building
    messages: Annotated[List[BaseMessage], add_messages]