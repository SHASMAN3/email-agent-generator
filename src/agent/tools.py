# src/agent/tools.py
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from langchain_core.tools import tool

# Load credentials from .env
load_dotenv()
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """
    A tool to send an email to a specified recipient using secure SMTP credentials.
    Returns SUCCESS or an ERROR message.
    """
    if not (SMTP_USERNAME and SMTP_APP_PASSWORD):
        return "ERROR: SMTP credentials missing in environment variables. Cannot send email."
        
    if "@" not in recipient:
        return "ERROR: Invalid recipient address format. Please check the recipient."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_USERNAME
    msg['To'] = recipient

    try:
        # Connect to the SMTP server and start TLS encryption
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_APP_PASSWORD)
            server.send_message(msg)
        
        print(f"\n--- Email Sent ---\nRecipient: {recipient}\nSubject: {subject}\n--------------------")
        return f"SUCCESS: Email titled '{subject}' sent to {recipient}."
    except smtplib.SMTPAuthenticationError:
        return "ERROR: SMTP Authentication failed. Check SMTP_USERNAME and SMTP_APP_PASSWORD."
    except Exception as e:
        return f"ERROR: Failed to send email via SMTP. Details: {str(e)}"
    
# List of tools available to the LangGraph agent
tools = [send_email]