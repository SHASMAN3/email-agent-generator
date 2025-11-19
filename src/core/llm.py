# src/core/llm.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.utils.function_calling import convert_to_openai_function

# Load environment variables
load_dotenv()

# Get tools for the LLM (which is now the send_email tool)
from src.agent.tools import tools

llm_with_tools = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
).bind(functions=[convert_to_openai_function(t) for t in tools])