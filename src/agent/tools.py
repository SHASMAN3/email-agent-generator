# src/agent/tools.py
import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field # <--- CRITICAL Pydantic Imports

# --- Pydantic Schema for Tool Input ---
class SendEmailArgsSchema(BaseModel):
    """Input parameters for the send_email tool."""
    recipient: str = Field(..., description="The full email address of the recipient.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The main text content of the email.")

# --- Core Email Sending Function ---

def send_email_func(
    recipient: str, 
    subject: str, 
    body: str, 
    username: str, 
    password: str, 
    host: str = "smtp.gmail.com", 
    port: int = 587
) -> str:
    """
    Core function to send an email using specified SMTP credentials.
    Returns SUCCESS or an ERROR message.
    """
    if not (username and password):
        return "ERROR: SMTP credentials missing. Cannot send email."
        
    if "@" not in recipient:
        return "ERROR: Invalid recipient address format. Please check the recipient."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = username
    msg['To'] = recipient

    try:
        # Connect to the SMTP server and start TLS encryption
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        
        print(f"\n--- Email Sent ---\nRecipient: {recipient}\nSubject: {subject}\n--------------------")
        return f"SUCCESS: Email titled '{subject}' sent to {recipient}."
    except smtplib.SMTPAuthenticationError:
        return "ERROR: SMTP Authentication failed. Check username and password."
    except Exception as e:
        return f"ERROR: Failed to send email via SMTP. Details: {str(e)}"

# --- Tool Definitions (Placeholder) ---

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    A placeholder tool. The real tool must be dynamically created in app.py 
    using send_email_func and user input credentials.
    """
    return "ERROR: Tool called without dynamic credentials. Check app.py setup."

# List of tools available to the LangGraph agent
tools = [send_email]