# src/core/llm.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
# Note: Removed unused imports like dotenv and convert_to_openai_function since they are now handled externally

def create_llm(api_key: str):
    """
    Initializes the ChatGoogleGenerativeAI model with a user-provided API key.

    Args:
        api_key: The user's Google/Gemini API key.

    Returns:
        A new instance of ChatGoogleGenerativeAI.
    """
    # The ChatGoogleGenerativeAI constructor automatically uses the 'google_api_key' 
    # parameter, ensuring it uses the user's key over a system environment variable.
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key 
    )

# Note: The global llm_with_tools definition has been removed. 
# This variable must now be defined dynamically in app.py after the user inputs 
# the API key and before the LangGraph agent is invoked.