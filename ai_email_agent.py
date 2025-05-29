#!/usr/bin/env python3
"""
AI Email Agent - A Python script for an AI agent that can respond to emails, check emails, and draft new emails.
"""

import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import json
import time
import logging
import requests
from getpass import getpass
from configparser import ConfigParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("email_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AI-Email-Agent")

class EmailConfig:
    """Handles email server configuration and authentication."""
    
    def __init__(self, config_file="email_config.ini"):
        """Initialize with configuration file or prompt for credentials."""
        self.config_file = config_file
        self.config = ConfigParser()
        
        # Default configuration
        self.imap_server = None
        self.smtp_server = None
        self.email_address = None
        self.password = None
        self.imap_port = 993
        self.smtp_port = 587
        
        # Load configuration if exists, otherwise prompt user
        if os.path.exists(config_file):
            self.load_config()
        else:
            self.setup_config()
    
    def load_config(self):
        """Load email configuration from file."""
        try:
            self.config.read(self.config_file)
            self.email_address = self.config.get('Email', 'email_address')
            self.password = self.config.get('Email', 'password')
            self.imap_server = self.config.get('Servers', 'imap_server')
            self.smtp_server = self.config.get('Servers', 'smtp_server')
            self.imap_port = self.config.getint('Servers', 'imap_port')
            self.smtp_port = self.config.getint('Servers', 'smtp_port')
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.setup_config()
    
    def setup_config(self):
        """Prompt user for email configuration and save to file."""
        print("Email Configuration Setup")
        self.email_address = input("Email address: ")
        self.password = getpass("Password: ")
        self.imap_server = input("IMAP server (e.g., imap.gmail.com): ")
        self.smtp_server = input("SMTP server (e.g., smtp.gmail.com): ")
        
        try:
            self.imap_port = int(input("IMAP port (default 993): ") or "993")
            self.smtp_port = int(input("SMTP port (default 587): ") or "587")
            
            # Save configuration
            self.config['Email'] = {
                'email_address': self.email_address,
                'password': self.password
            }
            self.config['Servers'] = {
                'imap_server': self.imap_server,
                'smtp_server': self.smtp_server,
                'imap_port': str(self.imap_port),
                'smtp_port': str(self.smtp_port)
            }
            
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error setting up configuration: {e}")


class EmailRetriever:
    """Handles retrieving and parsing emails from the server."""
    
    def __init__(self, config):
        """Initialize with email configuration."""
        self.config = config
        self.imap_connection = None
    
    def connect(self):
        """Connect to the IMAP server."""
        try:
            self.imap_connection = imaplib.IMAP4_SSL(
                self.config.imap_server, 
                self.config.imap_port
            )
            self.imap_connection.login(
                self.config.email_address, 
                self.config.password
            )
            logger.info("Connected to IMAP server successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to IMAP server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the IMAP server."""
        if self.imap_connection:
            try:
                self.imap_connection.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.error(f"Error disconnecting from IMAP server: {e}")
    
    def get_unread_emails(self, folder="INBOX", limit=10):
        """Retrieve unread emails from the specified folder."""
        if not self.imap_connection:
            if not self.connect():
                return []
        
        emails = []
        try:
            # Select the mailbox/folder
            status, messages = self.imap_connection.select(folder)
            if status != "OK":
                logger.error(f"Error selecting folder {folder}: {messages}")
                return []
            
            # Search for unread emails
            status, messages = self.imap_connection.search(None, 'UNSEEN')
            if status != "OK":
                logger.error("Error searching for unread emails")
                return []
            
            # Get the list of email IDs
            email_ids = messages[0].split()
            
            # Limit the number of emails to process
            email_ids = email_ids[-limit:] if limit > 0 else email_ids
            
            for email_id in email_ids:
                status, msg_data = self.imap_connection.fetch(email_id, '(RFC822)')
                if status != "OK":
                    logger.error(f"Error fetching email {email_id}")
                    continue
                
                raw_email = msg_data[0][1]
                email_message = email.message_from_bytes(raw_email)
                
                # Extract email details
                subject = self._decode_header(email_message["Subject"])
                sender = self._decode_header(email_message["From"])
                date = email_message["Date"]
                
                # Get email body
                body = self._get_email_body(email_message)
                
                emails.append({
                    "id": email_id.decode(),
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "body": body
                })
                
                logger.info(f"Retrieved email: {subject}")
            
            return emails
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            return []
    
    def _decode_header(self, header):
        """Decode email header."""
        if header is None:
            return ""
        
        try:
            decoded_header = email.header.decode_header(header)
            header_parts = []
            for part, encoding in decoded_header:
                if isinstance(part, bytes):
                    if encoding:
                        header_parts.append(part.decode(encoding or 'utf-8', errors='replace'))
                    else:
                        header_parts.append(part.decode('utf-8', errors='replace'))
                else:
                    header_parts.append(str(part))
            return " ".join(header_parts)
        except Exception as e:
            logger.error(f"Error decoding header: {e}")
            return str(header)
    
    def _get_email_body(self, email_message):
        """Extract the email body from the message."""
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Get the body text
                if content_type == "text/plain":
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        body = part.get_payload(decode=True).decode(charset, errors='replace')
                        break
                    except Exception as e:
                        logger.error(f"Error extracting plain text body: {e}")
                
                # If no plain text, try HTML
                elif content_type == "text/html" and not body:
                    try:
                        charset = part.get_content_charset() or 'utf-8'
                        body = part.get_payload(decode=True).decode(charset, errors='replace')
                        # Here you could add HTML to text conversion if needed
                    except Exception as e:
                        logger.error(f"Error extracting HTML body: {e}")
        else:
            # Not multipart - get the content directly
            content_type = email_message.get_content_type()
            try:
                charset = email_message.get_content_charset() or 'utf-8'
                body = email_message.get_payload(decode=True).decode(charset, errors='replace')
            except Exception as e:
                logger.error(f"Error extracting email body: {e}")
        
        return body


class AIResponseGenerator:
    """Handles generating AI responses to emails."""
    
    def __init__(self, api_key=None, api_url=None):
        """Initialize with API key and URL for the AI service."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No API key provided for AI service")
            self.api_key = input("Enter your OpenAI API key: ")
    
    def generate_response(self, email_data):
        """Generate an AI response based on the email content."""
        if not self.api_key:
            logger.error("No API key available for AI service")
            return None
        
        try:
            # Prepare the prompt for the AI
            prompt = self._create_prompt(email_data)
            
            # Call the OpenAI API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an email assistant. Generate a professional and helpful response to the following email."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                response_json = response.json()
                ai_response = response_json["choices"][0]["message"]["content"].strip()
                logger.info("AI response generated successfully")
                return ai_response
            else:
                logger.error(f"Error from AI service: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None
    
    def _create_prompt(self, email_data):
        """Create a prompt for the AI based on the email data."""
        return f"""
Please generate a response to the following email:

From: {email_data['sender']}
Subject: {email_data['subject']}
Date: {email_data['date']}

{email_data['body']}

Generate a professional and concise response that addresses the content of this email.
"""


class EmailSender:
    """Handles composing and sending emails."""
    
    def __init__(self, config):
        """Initialize with email configuration."""
        self.config = config
        self.smtp_connection = None
    
    def connect(self):
        """Connect to the SMTP server."""
        try:
            self.smtp_connection = smtplib.SMTP(
                self.config.smtp_server, 
                self.config.smtp_port
            )
            self.smtp_connection.ehlo()
            self.smtp_connection.starttls()
            self.smtp_connection.login(
                self.config.email_address, 
                self.config.password
            )
            logger.info("Connected to SMTP server successfully")
            return True
        except Exception as e:
            logger.error(f"Error connecting to SMTP server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the SMTP server."""
        if self.smtp_connection:
            try:
                self.smtp_connection.quit()
                logger.info("Disconnected from SMTP server")
            except Exception as e:
                logger.error(f"Error disconnecting from SMTP server: {e}")
    
    def send_email(self, to_address, subject, body, cc=None, bcc=None):
        """Send an email with the given details."""
        if not self.smtp_connection:
            if not self.connect():
                return False
        
        try:
            # Create a multipart message
            msg = MIMEMultipart()
            msg["From"] = self.config.email_address
            msg["To"] = to_address
            msg["Subject"] = subject
            
            if cc:
                msg["Cc"] = cc if isinstance(cc, str) else ", ".join(cc)
            if bcc:
                msg["Bcc"] = bcc if isinstance(bcc, str) else ", ".join(bcc)
            
            # Add body to email
            msg.attach(MIMEText(body, "plain"))
            
            # Get all recipients
            recipients = [to_address]
            if cc:
                recipients.extend(cc if isinstance(cc, list) else [cc])
            if bcc:
                recipients.extend(bcc if isinstance(bcc, list) else [bcc])
            
            # Send the email
            self.smtp_connection.sendmail(
                self.config.email_address,
                recipients,
                msg.as_string()
            )
            
            logger.info(f"Email sent to {to_address} with subject: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def reply_to_email(self, original_email, reply_body):
        """Reply to an email."""
        if not self.smtp_connection:
            if not self.connect():
                return False
        
        try:
            # Extract the sender's email address
            sender = original_email["sender"]
            # Simple extraction of email address from "Name <email@example.com>" format
            if "<" in sender and ">" in sender:
                to_address = sender.split("<")[1].split(">")[0].strip()
            else:
                to_address = sender.strip()
            
            # Create subject with Re: prefix if not already present
            subject = original_email["subject"]
            if not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
            
            # Send the reply
            return self.send_email(to_address, subject, reply_body)
        except Exception as e:
            logger.error(f"Error replying to email: {e}")
            return False


class AIEmailAgent:
    """Main class that orchestrates the email agent functionality."""
    
    def __init__(self):
        """Initialize the email agent components."""
        self.config = EmailConfig()
        self.retriever = EmailRetriever(self.config)
        self.ai = AIResponseGenerator()
        self.sender = EmailSender(self.config)
    
    def check_emails(self, folder="INBOX", limit=10):
        """Check for new emails and return them."""
        return self.retriever.get_unread_emails(folder, limit)
    
    def process_emails(self, auto_reply=False):
        """Process unread emails and optionally auto-reply."""
        emails = self.check_emails()
        
        if not emails:
            logger.info("No new emails to process")
            return []
        
        processed_emails = []
        for email_data in emails:
            # Generate AI response
            ai_response = self.ai.generate_response(email_data)
            
            if ai_response:
                email_data["ai_response"] = ai_response
                processed_emails.append(email_data)
                
                # Auto-reply if enabled
                if auto_reply:
                    self.sender.reply_to_email(email_data, ai_response)
                    logger.info(f"Auto-replied to email: {email_data['subject']}")
            else:
                logger.warning(f"Failed to generate AI response for email: {email_data['subject']}")
        
        return processed_emails
    
    def compose_new_email(self, to_address, subject, prompt, cc=None, bcc=None):
        """Compose a new email with AI-generated content based on a prompt."""
        # Create a mock email data structure for the AI
        email_data = {
            "sender": "me",
            "subject": f"Draft: {subject}",
            "date": time.strftime("%a, %d %b %Y %H:%M:%S"),
            "body": f"Please draft an email about: {prompt}"
        }
        
        # Generate the email content
        ai_content = self.ai.generate_response(email_data)
        
        if not ai_content:
            logger.error("Failed to generate email content")
            return False
        
        # Send the email
        return self.sender.send_email(to_address, subject, ai_content, cc, bcc)
    
    def draft_email(self, prompt):
        """Draft an email without sending it."""
        # Create a mock email data structure for the AI
        email_data = {
            "sender": "me",
            "subject": "Draft Email",
            "date": time.strftime("%a, %d %b %Y %H:%M:%S"),
            "body": f"Please draft an email about: {prompt}"
        }
        
        # Generate the email content
        ai_content = self.ai.generate_response(email_data)
        
        if not ai_content:
            logger.error("Failed to generate email draft")
            return None
        
        return ai_content
    
    def close_connections(self):
        """Close all server connections."""
        self.retriever.disconnect()
        self.sender.disconnect()
        logger.info("All connections closed")


def main():
    """Main function to run the AI Email Agent."""
    print("AI Email Agent")
    print("==============")
    
    agent = AIEmailAgent()
    
    while True:
        print("\nOptions:")
        print("1. Check emails")
        print("2. Process emails (generate AI responses)")
        print("3. Process emails with auto-reply")
        print("4. Compose new email")
        print("5. Draft email")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            emails = agent.check_emails()
            if emails:
                print(f"\nFound {len(emails)} unread emails:")
                for i, email_data in enumerate(emails, 1):
                    print(f"{i}. From: {email_data['sender']}")
                    print(f"   Subject: {email_data['subject']}")
                    print(f"   Date: {email_data['date']}")
                    print()
            else:
                print("\nNo unread emails found.")
        
        elif choice == "2":
            processed = agent.process_emails(auto_reply=False)
            if processed:
                print(f"\nProcessed {len(processed)} emails with AI responses:")
                for i, email_data in enumerate(processed, 1):
                    print(f"{i}. From: {email_data['sender']}")
                    print(f"   Subject: {email_data['subject']}")
                    print(f"   AI Response: {email_data['ai_response'][:100]}...")
                    print()
            else:
                print("\nNo emails were processed.")
        
        elif choice == "3":
            processed = agent.process_emails(auto_reply=True)
            if processed:
                print(f"\nAuto-replied to {len(processed)} emails.")
            else:
                print("\nNo emails were auto-replied to.")
        
        elif choice == "4":
            to_address = input("To: ")
            subject = input("Subject: ")
            prompt = input("What should the email be about? ")
            cc = input("CC (optional, separate multiple with commas): ")
            cc = [email.strip() for email in cc.split(",")] if cc else None
            
            if agent.compose_new_email(to_address, subject, prompt, cc):
                print("\nEmail composed and sent successfully!")
            else:
                print("\nFailed to compose and send email.")
        
        elif choice == "5":
            prompt = input("What should the email be about? ")
            draft = agent.draft_email(prompt)
            
            if draft:
                print("\nEmail Draft:")
                print("============")
                print(draft)
                print("============")
                
                save = input("\nSave draft to file? (y/n): ")
                if save.lower() == "y":
                    filename = input("Enter filename (default: email_draft.txt): ") or "email_draft.txt"
                    try:
                        with open(filename, "w") as f:
                            f.write(draft)
                        print(f"Draft saved to {filename}")
                    except Exception as e:
                        print(f"Error saving draft: {e}")
            else:
                print("\nFailed to generate email draft.")
        
        elif choice == "6":
            agent.close_connections()
            print("\nThank you for using AI Email Agent. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()