#!/usr/bin/env python3
"""
Enhanced AI Email Agent - A Python script for an AI agent that can respond to emails, check emails, and draft new emails.
With improved error handling, security, performance, and fault tolerance.
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
import socket
import re
import base64
import argparse
import traceback
import requests
from email.header import decode_header
from getpass import getpass
from configparser import ConfigParser
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Tuple

# Try to import cryptography, but handle if not available
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

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
    """Handles email server configuration and authentication with enhanced security and validation."""
    
    # Define constants and regex patterns for validation
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    SERVER_REGEX = re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PORT_RANGE = (1, 65535)
    MAX_RETRIES = 3
    SECRET_KEY_ENV = "EMAIL_AGENT_SECRET_KEY"
    
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
        
        # Setup encryption key
        self._setup_encryption_key()
        
        # Load configuration if exists, otherwise prompt user
        if os.path.exists(config_file):
            self.load_config()
        else:
            self.setup_config()
    
    def _setup_encryption_key(self):
        """Set up encryption key for securing sensitive data."""
        self.secret_key = os.environ.get(self.SECRET_KEY_ENV)
        
        if not self.secret_key:
            # Generate a new key if not found in environment
            if CRYPTOGRAPHY_AVAILABLE:
                salt = b'email_agent_salt'  # In production, use a secure random salt
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                password = "default_password".encode()  # In production, prompt for a password
                self.secret_key = base64.urlsafe_b64encode(kdf.derive(password))
                os.environ[self.SECRET_KEY_ENV] = self.secret_key.decode()
                logger.info("Generated new encryption key")
            else:
                # Simple fallback if cryptography is not available
                self.secret_key = base64.b64encode(b"email_agent_default_key").decode()
                os.environ[self.SECRET_KEY_ENV] = self.secret_key
                logger.warning("Cryptography package not available, using simple encoding (less secure)")
        
        # Create Fernet cipher for encryption/decryption
        try:
            if CRYPTOGRAPHY_AVAILABLE:
                self.cipher = Fernet(self.secret_key if isinstance(self.secret_key, bytes) 
                                    else self.secret_key.encode())
                logger.debug("Encryption setup complete")
            else:
                self.cipher = None
        except Exception as e:
            logger.error(f"Error setting up encryption: {e}")
            # Fallback to a simpler encryption if Fernet fails
            self.cipher = None
            logger.warning("Falling back to base64 encoding (less secure)")
    
    def _encrypt(self, data):
        """Encrypt sensitive data."""
        if not data:
            return data
            
        try:
            if CRYPTOGRAPHY_AVAILABLE and self.cipher:
                return self.cipher.encrypt(data.encode()).decode()
            else:
                # Fallback to base64 (not secure, just obfuscation)
                return base64.b64encode(data.encode()).decode()
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            # Return plaintext if encryption fails, but log the error
            return data
    
    def _decrypt(self, data):
        """Decrypt sensitive data."""
        if not data:
            return data
            
        try:
            if CRYPTOGRAPHY_AVAILABLE and self.cipher:
                return self.cipher.decrypt(data.encode()).decode()
            else:
                # Fallback from base64
                return base64.b64decode(data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            # Return encrypted data if decryption fails
            return data
    
    def validate_email(self, email):
        """Validate email address format."""
        if not email or not isinstance(email, str):
            logger.error("Email address cannot be empty")
            return False
            
        if not self.EMAIL_REGEX.match(email):
            logger.error(f"Invalid email format: {email}")
            return False
            
        return True
    
    def validate_server(self, server):
        """Validate server address format."""
        if not server or not isinstance(server, str):
            logger.error("Server address cannot be empty")
            return False
            
        if not self.SERVER_REGEX.match(server):
            logger.error(f"Invalid server format: {server}")
            return False
            
        return True
    
    def validate_port(self, port):
        """Validate port number."""
        try:
            port = int(port)
            if port < self.PORT_RANGE[0] or port > self.PORT_RANGE[1]:
                logger.error(f"Port number out of range: {port}")
                return False
            return True
        except (ValueError, TypeError):
            logger.error(f"Invalid port number: {port}")
            return False
    
    def load_config(self):
        """Load email configuration from file with decryption."""
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                self.config.read(self.config_file)
                
                # Get encrypted values
                encrypted_email = self.config.get('Email', 'email_address')
                encrypted_password = self.config.get('Email', 'password')
                
                # Decrypt values
                self.email_address = encrypted_email  # Email is not encrypted
                self.password = self._decrypt(encrypted_password)
                
                # Get server information
                self.imap_server = self.config.get('Servers', 'imap_server')
                self.smtp_server = self.config.get('Servers', 'smtp_server')
                self.imap_port = self.config.getint('Servers', 'imap_port')
                self.smtp_port = self.config.getint('Servers', 'smtp_port')
                
                logger.info("Configuration loaded successfully")
                return True
            except Exception as e:
                retry_count += 1
                logger.error(f"Error loading configuration (attempt {retry_count}/{self.MAX_RETRIES}): {e}")
                if retry_count >= self.MAX_RETRIES:
                    logger.critical("Failed to load configuration after multiple attempts")
                    self.setup_config()
                    return False
                # Wait before retrying
                time.sleep(1)
    
    def setup_config(self):
        """Prompt user for email configuration and save to file with encryption."""
        print("Email Configuration Setup")
        
        # Get and validate email
        while True:
            self.email_address = input("Email address: ")
            if self.validate_email(self.email_address):
                break
            print("Invalid email format. Please try again.")
        
        # Get password
        self.password = getpass("Password: ")
        
        # Get and validate IMAP server
        while True:
            self.imap_server = input("IMAP server (e.g., imap.gmail.com): ")
            if self.validate_server(self.imap_server):
                break
            print("Invalid server format. Please try again.")
        
        # Get and validate SMTP server
        while True:
            self.smtp_server = input("SMTP server (e.g., smtp.gmail.com): ")
            if self.validate_server(self.smtp_server):
                break
            print("Invalid server format. Please try again.")
        
        # Get and validate ports
        while True:
            imap_port_input = input("IMAP port (default 993): ") or "993"
            if self.validate_port(imap_port_input):
                self.imap_port = int(imap_port_input)
                break
            print("Invalid port number. Please try again.")
        
        while True:
            smtp_port_input = input("SMTP port (default 587): ") or "587"
            if self.validate_port(smtp_port_input):
                self.smtp_port = int(smtp_port_input)
                break
            print("Invalid port number. Please try again.")
        
        try:
            # Encrypt sensitive data
            encrypted_password = self._encrypt(self.password)
            
            # Save configuration
            self.config['Email'] = {
                'email_address': self.email_address,
                'password': encrypted_password
            }
            self.config['Servers'] = {
                'imap_server': self.imap_server,
                'smtp_server': self.smtp_server,
                'imap_port': str(self.imap_port),
                'smtp_port': str(self.smtp_port)
            }
            
            # Set secure permissions for the config file
            if os.path.exists(self.config_file):
                try:
                    os.chmod(self.config_file, 0o600)  # Read/write for owner only
                except Exception as e:
                    logger.warning(f"Could not set secure permissions on config file: {e}")
                
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            # Set secure permissions again after writing
            try:
                os.chmod(self.config_file, 0o600)
            except Exception as e:
                logger.warning(f"Could not set secure permissions on config file: {e}")
            
            logger.info("Configuration saved successfully with encryption")
            return True
        except Exception as e:
            logger.error(f"Error setting up configuration: {e}")
            return False
    
    def test_connection(self):
        """Test connection to email servers to validate configuration."""
        imap_success = False
        smtp_success = False
        
        try:
            # Test IMAP connection
            imap_conn = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            imap_conn.login(self.email_address, self.password)
            imap_conn.logout()
            imap_success = True
            logger.info("IMAP connection test successful")
        except Exception as e:
            logger.error(f"IMAP connection test failed: {e}")
        
        try:
            # Test SMTP connection
            smtp_conn = smtplib.SMTP(self.smtp_server, self.smtp_port)
            smtp_conn.ehlo()
            smtp_conn.starttls()
            smtp_conn.login(self.email_address, self.password)
            smtp_conn.quit()
            smtp_success = True
            logger.info("SMTP connection test successful")
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
        
        return {
            "imap": imap_success,
            "smtp": smtp_success,
            "overall": imap_success and smtp_success
        }


class EmailRetriever:
    """Handles retrieving and parsing emails from the server with enhanced error handling and performance."""
    
    # Define constants for retry logic and performance
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BACKOFF_FACTOR = 2
    CONNECTION_TIMEOUT = 30  # seconds
    CACHE_SIZE = 100
    
    def __init__(self, config):
        """Initialize with email configuration."""
        self.config = config
        self.imap_connection = None
        self.connected = False
        self.last_error = None
        self.performance_metrics = {
            "connection_time": 0,
            "fetch_time": 0,
            "parse_time": 0,
            "total_emails_processed": 0
        }
    
    def connect(self) -> bool:
        """Connect to the IMAP server with retry logic."""
        if self.connected and self.imap_connection:
            try:
                # Check if connection is still alive
                status, _ = self.imap_connection.noop()
                if status == "OK":
                    logger.debug("Reusing existing IMAP connection")
                    return True
            except Exception:
                # Connection is stale, close it
                self.disconnect()
        
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                start_time = time.time()
                
                # Set socket timeout
                socket.setdefaulttimeout(self.CONNECTION_TIMEOUT)
                
                self.imap_connection = imaplib.IMAP4_SSL(
                    self.config.imap_server, 
                    self.config.imap_port
                )
                
                self.imap_connection.login(
                    self.config.email_address, 
                    self.config.password
                )
                
                connection_time = time.time() - start_time
                self.performance_metrics["connection_time"] = connection_time
                
                logger.info(f"Connected to IMAP server successfully in {connection_time:.2f}s")
                self.connected = True
                self.last_error = None
                return True
                
            except imaplib.IMAP4.error as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"IMAP authentication error: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                
            except (socket.gaierror, socket.timeout) as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Network error connecting to IMAP server: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Unexpected error connecting to IMAP server: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
        
        logger.critical(f"Failed to connect to IMAP server after {self.MAX_RETRIES} attempts")
        self.connected = False
        return False
    
    def disconnect(self) -> None:
        """Disconnect from the IMAP server safely."""
        if self.imap_connection:
            try:
                if self.connected:
                    self.imap_connection.logout()
                else:
                    self.imap_connection.close()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.error(f"Error disconnecting from IMAP server: {e}")
            finally:
                self.imap_connection = None
                self.connected = False
    
    def get_unread_emails(self, folder: str = "INBOX", limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve unread emails from the specified folder with enhanced error handling."""
        if not self.connected:
            if not self.connect():
                logger.error("Cannot retrieve emails: Not connected to IMAP server")
                return []
        
        emails = []
        start_time = time.time()
        
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
            email_ids = email_ids[-limit:] if limit > 0 and len(email_ids) > limit else email_ids
            
            if not email_ids:
                logger.info("No unread emails found")
                return []
                
            logger.info(f"Found {len(email_ids)} unread emails, processing up to {limit}")
            
            fetch_start_time = time.time()
            for email_id in email_ids:
                try:
                    # Fetch email with retry logic
                    email_data = self._fetch_email_with_retry(email_id)
                    if email_data:
                        emails.append(email_data)
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
            
            fetch_time = time.time() - fetch_start_time
            self.performance_metrics["fetch_time"] = fetch_time
            self.performance_metrics["total_emails_processed"] += len(emails)
            
            total_time = time.time() - start_time
            logger.info(f"Retrieved {len(emails)} emails in {total_time:.2f}s (fetch: {fetch_time:.2f}s)")
            
            return emails
            
        except imaplib.IMAP4.error as e:
            self.last_error = f"IMAP error retrieving emails: {e}"
            logger.error(self.last_error)
            # Try to reconnect on next operation
            self.disconnect()
            return []
            
        except Exception as e:
            self.last_error = f"Unexpected error retrieving emails: {e}"
            logger.error(self.last_error)
            # Try to reconnect on next operation
            self.disconnect()
            return []
    
    def _fetch_email_with_retry(self, email_id) -> Optional[Dict[str, Any]]:
        """Fetch a single email with retry logic."""
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                status, msg_data = self.imap_connection.fetch(email_id, '(RFC822)')
                if status != "OK":
                    logger.error(f"Error fetching email {email_id}: {msg_data}")
                    retry_count += 1
                    time.sleep(self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1)))
                    continue
                
                if not msg_data or not msg_data[0]:
                    logger.error(f"No data returned for email {email_id}")
                    return None
                
                raw_email = msg_data[0][1]
                parse_start_time = time.time()
                email_message = email.message_from_bytes(raw_email)
                
                # Extract email details
                subject = self._decode_header(email_message["Subject"])
                sender = self._decode_header(email_message["From"])
                date = email_message["Date"]
                message_id = email_message["Message-ID"]
                
                # Get email body
                body = self._get_email_body(email_message)
                
                parse_time = time.time() - parse_start_time
                self.performance_metrics["parse_time"] += parse_time
                
                email_data = {
                    "id": email_id.decode(),
                    "message_id": message_id,
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "body": body,
                    "raw_size": len(raw_email),
                    "parse_time": parse_time
                }
                
                logger.info(f"Retrieved email: {subject} ({len(raw_email)} bytes)")
                return email_data
                
            except Exception as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                logger.error(f"Error fetching email {email_id} (attempt {retry_count}/{self.MAX_RETRIES}): {e}")
                time.sleep(delay)
        
        logger.error(f"Failed to fetch email {email_id} after {self.MAX_RETRIES} attempts")
        return None
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _decode_header(self, header) -> str:
        """Decode email header with caching for performance."""
        if header is None:
            return ""
        
        try:
            decoded_header = decode_header(header)
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
    
    def _get_email_body(self, email_message) -> str:
        """Extract the email body from the message with improved error handling."""
        body = ""
        
        try:
            if email_message.is_multipart():
                # First try to find plain text
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition") or "")
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Get the body text
                    if content_type == "text/plain":
                        try:
                            charset = part.get_content_charset() or 'utf-8'
                            part_body = part.get_payload(decode=True)
                            if part_body:
                                body = part_body.decode(charset, errors='replace')
                                break
                        except Exception as e:
                            logger.error(f"Error extracting plain text body: {e}")
                
                # If no plain text, try HTML
                if not body:
                    for part in email_message.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition") or "")
                        
                        # Skip attachments
                        if "attachment" in content_disposition:
                            continue
                        
                        if content_type == "text/html":
                            try:
                                charset = part.get_content_charset() or 'utf-8'
                                part_body = part.get_payload(decode=True)
                                if part_body:
                                    body = part_body.decode(charset, errors='replace')
                                    # Here you could add HTML to text conversion if needed
                                    break
                            except Exception as e:
                                logger.error(f"Error extracting HTML body: {e}")
            else:
                # Not multipart - get the content directly
                content_type = email_message.get_content_type()
                try:
                    charset = email_message.get_content_charset() or 'utf-8'
                    part_body = email_message.get_payload(decode=True)
                    if part_body:
                        body = part_body.decode(charset, errors='replace')
                except Exception as e:
                    logger.error(f"Error extracting email body: {e}")
            
            return body
        except Exception as e:
            logger.error(f"Unexpected error extracting email body: {e}")
            return f"[Error extracting email body: {e}]"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        return self.performance_metrics
    
    def mark_as_read(self, email_id: str) -> bool:
        """Mark an email as read."""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Convert string ID to bytes if needed
            if isinstance(email_id, str):
                email_id = email_id.encode()
                
            # Add the \Seen flag
            self.imap_connection.store(email_id, '+FLAGS', '\\Seen')
            logger.info(f"Marked email {email_id} as read")
            return True
        except Exception as e:
            logger.error(f"Error marking email {email_id} as read: {e}")
            return False
    
    def get_folders(self) -> List[str]:
        """Get available folders/mailboxes."""
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            status, folder_list = self.imap_connection.list()
            if status != "OK":
                logger.error(f"Error getting folders: {folder_list}")
                return []
            
            folders = []
            for folder in folder_list:
                if folder:
                    # Parse folder name from response
                    parts = folder.decode().split(' "." ')
                    if len(parts) > 1:
                        folder_name = parts[1].strip('"')
                        folders.append(folder_name)
            
            return folders
        except Exception as e:
            logger.error(f"Error getting folders: {e}")
            return []


class AIResponseGenerator:
    """Handles generating AI responses to emails with enhanced error handling and fallbacks."""
    
    # Define constants for API interaction and performance
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BACKOFF_FACTOR = 2
    REQUEST_TIMEOUT = 30  # seconds
    CACHE_SIZE = 50
    DEFAULT_MODEL = "gpt-3.5-turbo"
    FALLBACK_MODEL = "gpt-3.5-turbo-instruct"  # Fallback to a different model if primary fails
    
    def __init__(self, api_key=None, api_url=None):
        """Initialize with API key and URL for the AI service."""
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = api_url or "https://api.openai.com/v1/chat/completions"
        self.fallback_api_url = "https://api.openai.com/v1/completions"  # Fallback API endpoint
        
        self.last_error = None
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_count": 0,
            "fallback_count": 0,
            "total_tokens": 0,
            "average_response_time": 0,
            "total_response_time": 0
        }
        
        if not self.api_key:
            logger.warning("No API key provided for AI service")
            self.api_key = input("Enter your OpenAI API key: ")
            # Store in environment variable for future use
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def generate_response(self, email_data: Dict[str, Any], max_tokens: int = 500) -> Optional[str]:
        """Generate an AI response based on the email content with retry and fallback."""
        if not self.api_key:
            self.last_error = "No API key available for AI service"
            logger.error(self.last_error)
            return self._get_fallback_response("no_api_key", email_data)
        
        # Validate input
        if not email_data or not isinstance(email_data, dict):
            self.last_error = "Invalid email data provided"
            logger.error(self.last_error)
            return self._get_fallback_response("invalid_input", email_data)
        
        # Check for required fields
        required_fields = ["subject", "sender", "body"]
        for field in required_fields:
            if field not in email_data or not email_data[field]:
                self.last_error = f"Missing required field in email data: {field}"
                logger.error(self.last_error)
                return self._get_fallback_response("missing_field", email_data, missing_field=field)
        
        # Update metrics
        self.performance_metrics["total_requests"] += 1
        
        # Try primary model first
        ai_response = self._call_openai_api(email_data, max_tokens)
        
        # If primary model fails, try fallback
        if not ai_response:
            logger.warning("Primary model failed, trying fallback model")
            self.performance_metrics["fallback_count"] += 1
            ai_response = self._call_openai_api_fallback(email_data, max_tokens)
        
        # If both fail, use template response
        if not ai_response:
            logger.error("Both primary and fallback models failed")
            return self._get_fallback_response("api_failure", email_data)
        
        return ai_response
    
    def _call_openai_api(self, email_data: Dict[str, Any], max_tokens: int) -> Optional[str]:
        """Call the OpenAI API with retry logic."""
        retry_count = 0
        start_time = time.time()
        
        while retry_count < self.MAX_RETRIES:
            try:
                # Prepare the prompt for the AI
                prompt = self._create_prompt(email_data)
                
                # Call the OpenAI API
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": self.DEFAULT_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are an email assistant. Generate a professional and helpful response to the following email."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=self.REQUEST_TIMEOUT
                )
                
                # Update response time metrics
                response_time = time.time() - start_time
                self.performance_metrics["total_response_time"] += response_time
                self.performance_metrics["average_response_time"] = (
                    self.performance_metrics["total_response_time"] / 
                    self.performance_metrics["total_requests"]
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    ai_response = response_json["choices"][0]["message"]["content"].strip()
                    
                    # Update metrics
                    self.performance_metrics["successful_requests"] += 1
                    if "usage" in response_json:
                        self.performance_metrics["total_tokens"] += response_json["usage"]["total_tokens"]
                    
                    logger.info(f"AI response generated successfully in {response_time:.2f}s")
                    return ai_response
                else:
                    error_msg = f"Error from AI service: {response.status_code} - {response.text}"
                    self.last_error = error_msg
                    logger.error(error_msg)
                    
                    # Check if we should retry based on error code
                    if response.status_code in [429, 500, 502, 503, 504]:
                        retry_count += 1
                        self.performance_metrics["retry_count"] += 1
                        delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                        logger.warning(f"Retrying in {delay}s (attempt {retry_count}/{self.MAX_RETRIES})")
                        time.sleep(delay)
                    else:
                        # Don't retry for client errors except rate limits
                        self.performance_metrics["failed_requests"] += 1
                        break
            
            except requests.exceptions.Timeout:
                retry_count += 1
                self.performance_metrics["retry_count"] += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Timeout calling OpenAI API"
                logger.error(f"{self.last_error}. Retrying in {delay}s (attempt {retry_count}/{self.MAX_RETRIES})")
                time.sleep(delay)
            
            except requests.exceptions.RequestException as e:
                retry_count += 1
                self.performance_metrics["retry_count"] += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Request error calling OpenAI API: {e}"
                logger.error(f"{self.last_error}. Retrying in {delay}s (attempt {retry_count}/{self.MAX_RETRIES})")
                time.sleep(delay)
            
            except Exception as e:
                retry_count += 1
                self.performance_metrics["retry_count"] += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Unexpected error generating AI response: {e}"
                logger.error(f"{self.last_error}. Retrying in {delay}s (attempt {retry_count}/{self.MAX_RETRIES})")
                time.sleep(delay)
        
        # All retries failed
        self.performance_metrics["failed_requests"] += 1
        logger.error(f"Failed to generate AI response after {self.MAX_RETRIES} attempts")
        return None
    
    def _call_openai_api_fallback(self, email_data: Dict[str, Any], max_tokens: int) -> Optional[str]:
        """Call the OpenAI API with fallback model."""
        try:
            # Prepare the prompt for the fallback AI model
            prompt = self._create_prompt(email_data)
            
            # Call the OpenAI API with fallback model
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Use completions API instead of chat API for fallback
            data = {
                "model": self.FALLBACK_MODEL,
                "prompt": f"You are an email assistant. Generate a professional and helpful response to the following email:\n\n{prompt}",
                "temperature": 0.7,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                self.fallback_api_url,
                headers=headers,
                json=data,
                timeout=self.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                response_json = response.json()
                ai_response = response_json["choices"][0]["text"].strip()
                
                # Update metrics
                self.performance_metrics["successful_requests"] += 1
                if "usage" in response_json:
                    self.performance_metrics["total_tokens"] += response_json["usage"]["total_tokens"]
                
                logger.info("AI response generated successfully using fallback model")
                return ai_response
            else:
                error_msg = f"Error from fallback AI service: {response.status_code} - {response.text}"
                self.last_error = error_msg
                logger.error(error_msg)
                return None
                
        except Exception as e:
            self.last_error = f"Unexpected error generating fallback AI response: {e}"
            logger.error(self.last_error)
            return None
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _create_prompt(self, email_data: Dict[str, Any]) -> str:
        """Create a prompt for the AI based on the email data with caching for repeated requests."""
        # Format the prompt with all available email data
        sender = email_data.get('sender', 'Unknown Sender')
        subject = email_data.get('subject', 'No Subject')
        date = email_data.get('date', 'Unknown Date')
        body = email_data.get('body', 'No Content')
        
        return f"""
Please generate a response to the following email:

From: {sender}
Subject: {subject}
Date: {date}

{body}

Generate a professional and concise response that addresses the content of this email.
"""
    
    def _get_fallback_response(self, error_type: str, email_data: Dict[str, Any], **kwargs) -> str:
        """Generate a fallback response when AI generation fails."""
        sender_name = "there"
        if "sender" in email_data and email_data["sender"]:
            # Extract name from email format "Name <email@example.com>"
            sender = email_data["sender"]
            if "<" in sender:
                sender_name = sender.split("<")[0].strip()
            else:
                sender_name = sender.split("@")[0].strip()
        
        subject = email_data.get("subject", "your email")
        
        templates = {
            "no_api_key": f"""
Hello {sender_name},

Thank you for your email regarding "{subject}". I've received your message and will respond properly as soon as possible.

Best regards,
[Your Name]
""",
            "invalid_input": f"""
Hello {sender_name},

Thank you for your email. I've received your message and will get back to you with a detailed response shortly.

Best regards,
[Your Name]
""",
            "missing_field": f"""
Hello {sender_name},

Thank you for your email regarding "{subject}". I've received your message and will respond to your inquiry soon.

Best regards,
[Your Name]
""",
            "api_failure": f"""
Hello {sender_name},

Thank you for your email regarding "{subject}". I've received your message and will review it carefully. I'll get back to you with a comprehensive response as soon as possible.

Best regards,
[Your Name]
"""
        }
        
        response = templates.get(error_type, templates["api_failure"])
        logger.info(f"Generated fallback response for error type: {error_type}")
        return response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        return self.performance_metrics


class EmailSender:
    """Handles composing and sending emails with enhanced error handling and security."""
    
    # Define constants for retry logic and performance
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BACKOFF_FACTOR = 2
    CONNECTION_TIMEOUT = 30  # seconds
    
    def __init__(self, config):
        """Initialize with email configuration."""
        self.config = config
        self.smtp_connection = None
        self.connected = False
        self.last_error = None
        self.performance_metrics = {
            "connection_time": 0,
            "send_time": 0,
            "total_emails_sent": 0,
            "failed_sends": 0
        }
    
    def connect(self) -> bool:
        """Connect to the SMTP server with retry logic."""
        if self.connected and self.smtp_connection:
            try:
                # Check if connection is still alive
                status = self.smtp_connection.noop()[0]
                if status == 250:
                    logger.debug("Reusing existing SMTP connection")
                    return True
            except Exception:
                # Connection is stale, close it
                self.disconnect()
        
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                start_time = time.time()
                
                # Set socket timeout
                socket.setdefaulttimeout(self.CONNECTION_TIMEOUT)
                
                self.smtp_connection = smtplib.SMTP(
                    self.config.smtp_server, 
                    self.config.smtp_port
                )
                
                # Identify ourselves to the server
                self.smtp_connection.ehlo()
                
                # If we can encrypt the connection, do it
                if self.smtp_connection.has_extn('STARTTLS'):
                    self.smtp_connection.starttls()
                    self.smtp_connection.ehlo()  # Re-identify ourselves over TLS
                else:
                    logger.warning("SMTP server does not support STARTTLS. Connection is not encrypted.")
                
                # Login to the server
                self.smtp_connection.login(
                    self.config.email_address, 
                    self.config.password
                )
                
                connection_time = time.time() - start_time
                self.performance_metrics["connection_time"] = connection_time
                
                logger.info(f"Connected to SMTP server successfully in {connection_time:.2f}s")
                self.connected = True
                self.last_error = None
                return True
                
            except smtplib.SMTPAuthenticationError as e:
                self.last_error = f"SMTP authentication error: {e}"
                logger.error(self.last_error)
                # Don't retry auth errors
                break
                
            except smtplib.SMTPException as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"SMTP error: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                
            except (socket.gaierror, socket.timeout) as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Network error connecting to SMTP server: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Unexpected error connecting to SMTP server: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
        
        logger.critical(f"Failed to connect to SMTP server after {self.MAX_RETRIES} attempts")
        self.connected = False
        return False
    
    def disconnect(self) -> None:
        """Disconnect from the SMTP server safely."""
        if self.smtp_connection:
            try:
                self.smtp_connection.quit()
                logger.info("Disconnected from SMTP server")
            except Exception as e:
                logger.error(f"Error disconnecting from SMTP server: {e}")
                try:
                    self.smtp_connection.close()
                except Exception:
                    pass
            finally:
                self.smtp_connection = None
                self.connected = False
    
    def send_email(self, to_address: str, subject: str, body: str, 
                  cc: Optional[Union[str, List[str]]] = None, 
                  bcc: Optional[Union[str, List[str]]] = None) -> bool:
        """Send an email with the given details with retry logic."""
        # Validate inputs
        if not to_address or not isinstance(to_address, str):
            logger.error("Invalid 'to' address")
            return False
        
        if not subject:
            logger.warning("Email has no subject")
            subject = "(No Subject)"
        
        if not body:
            logger.warning("Email has no body")
            body = "(No Content)"
        
        # Connect if not already connected
        if not self.connected:
            if not self.connect():
                return False
        
        # Process cc and bcc
        cc_list = self._process_address_list(cc)
        bcc_list = self._process_address_list(bcc)
        
        # Create a multipart message
        msg = MIMEMultipart()
        msg["From"] = self.config.email_address
        msg["To"] = to_address
        msg["Subject"] = subject
        
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        
        # Add body to email
        msg.attach(MIMEText(body, "plain"))
        
        # Get all recipients
        recipients = [to_address] + cc_list + bcc_list
        
        # Send the email with retry logic
        retry_count = 0
        while retry_count < self.MAX_RETRIES:
            try:
                start_time = time.time()
                
                self.smtp_connection.sendmail(
                    self.config.email_address,
                    recipients,
                    msg.as_string()
                )
                
                send_time = time.time() - start_time
                self.performance_metrics["send_time"] += send_time
                self.performance_metrics["total_emails_sent"] += 1
                
                logger.info(f"Email sent to {to_address} with subject: {subject} in {send_time:.2f}s")
                return True
                
            except smtplib.SMTPServerDisconnected:
                retry_count += 1
                logger.warning(f"SMTP server disconnected. Reconnecting (attempt {retry_count}/{self.MAX_RETRIES})")
                self.connected = False
                if not self.connect():
                    continue
                    
            except smtplib.SMTPException as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"SMTP error sending email: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                
            except Exception as e:
                retry_count += 1
                delay = self.RETRY_DELAY * (self.BACKOFF_FACTOR ** (retry_count - 1))
                self.last_error = f"Unexpected error sending email: {e}"
                logger.error(f"{self.last_error}. Retry {retry_count}/{self.MAX_RETRIES} in {delay}s")
                time.sleep(delay)
        
        # All retries failed
        logger.error(f"Failed to send email to {to_address} after {self.MAX_RETRIES} attempts")
        self.performance_metrics["failed_sends"] += 1
        return False
    
    def _process_address_list(self, addresses: Optional[Union[str, List[str]]]) -> List[str]:
        """Process and validate email address list."""
        result = []
        
        if not addresses:
            return result
            
        # Convert string to list if needed
        if isinstance(addresses, str):
            # Split by comma if it's a comma-separated list
            if "," in addresses:
                addresses = [addr.strip() for addr in addresses.split(",")]
            else:
                addresses = [addresses]
        
        # Validate each address
        for addr in addresses:
            if addr and isinstance(addr, str) and "@" in addr:
                result.append(addr)
            else:
                logger.warning(f"Invalid email address skipped: {addr}")
        
        return result
    
    def reply_to_email(self, original_email: Dict[str, Any], reply_body: str) -> bool:
        """Reply to an email with enhanced error handling."""
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Extract the sender's email address
            sender = original_email.get("sender", "")
            if not sender:
                logger.error("Cannot reply: original email has no sender")
                return False
                
            # Simple extraction of email address from "Name <email@example.com>" format
            to_address = ""
            if "<" in sender and ">" in sender:
                to_address = sender.split("<")[1].split(">")[0].strip()
            else:
                to_address = sender.strip()
            
            if not to_address or "@" not in to_address:
                logger.error(f"Cannot reply: invalid sender address format: {sender}")
                return False
            
            # Create subject with Re: prefix if not already present
            subject = original_email.get("subject", "")
            if not subject:
                subject = "Re: (No Subject)"
            elif not subject.lower().startswith("re:"):
                subject = f"Re: {subject}"
            
            # Add original message to the reply
            full_reply = reply_body
            
            if original_email.get("body"):
                full_reply += f"\n\n---------- Original Message ----------\n"
                full_reply += f"From: {sender}\n"
                full_reply += f"Date: {original_email.get('date', 'Unknown')}\n"
                full_reply += f"Subject: {original_email.get('subject', 'No Subject')}\n\n"
                
                # Add the original message with > prefix for each line
                original_body = original_email.get("body", "")
                quoted_body = "\n".join([f"> {line}" for line in original_body.split("\n")])
                full_reply += quoted_body
            
            # Send the reply
            return self.send_email(to_address, subject, full_reply)
            
        except Exception as e:
            self.last_error = f"Error replying to email: {e}"
            logger.error(self.last_error)
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring."""
        return self.performance_metrics


class AIEmailAgent:
    """Main class that orchestrates the email agent functionality with enhanced reliability."""
    
    # Define constants for performance and reliability
    MAX_RETRIES = 3
    HEALTH_CHECK_INTERVAL = 300  # seconds (5 minutes)
    
    def __init__(self):
        """Initialize the email agent components with error handling."""
        try:
            logger.info("Initializing AI Email Agent")
            self.config = EmailConfig()
            self.retriever = EmailRetriever(self.config)
            self.ai = AIResponseGenerator()
            self.sender = EmailSender(self.config)
            
            self.last_health_check = 0
            self.health_status = {
                "config": True,
                "retriever": False,
                "ai": False,
                "sender": False,
                "overall": False
            }
            
            self.performance_metrics = {
                "start_time": time.time(),
                "emails_checked": 0,
                "emails_processed": 0,
                "emails_replied": 0,
                "emails_composed": 0,
                "emails_drafted": 0,
                "errors": 0
            }
            
            # Perform initial health check
            self._check_health()
            
            logger.info("AI Email Agent initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize AI Email Agent: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    def _check_health(self) -> Dict[str, bool]:
        """Check the health of all components."""
        current_time = time.time()
        
        # Only run health check if it's been more than HEALTH_CHECK_INTERVAL since last check
        if current_time - self.last_health_check < self.HEALTH_CHECK_INTERVAL:
            return self.health_status
            
        logger.info("Performing health check")
        self.last_health_check = current_time
        
        # Check retriever
        retriever_health = False
        try:
            if self.retriever.connect():
                retriever_health = True
                self.retriever.disconnect()
        except Exception as e:
            logger.error(f"Retriever health check failed: {e}")
        
        # Check sender
        sender_health = False
        try:
            if self.sender.connect():
                sender_health = True
                self.sender.disconnect()
        except Exception as e:
            logger.error(f"Sender health check failed: {e}")
        
        # Check AI (simple check if API key exists)
        ai_health = bool(self.ai.api_key)
        
        # Update health status
        self.health_status = {
            "config": True,  # Config is always available once loaded
            "retriever": retriever_health,
            "ai": ai_health,
            "sender": sender_health,
            "overall": retriever_health and ai_health and sender_health
        }
        
        logger.info(f"Health check results: {json.dumps(self.health_status)}")
        return self.health_status
    
    def check_emails(self, folder: str = "INBOX", limit: int = 10) -> List[Dict[str, Any]]:
        """Check for new emails and return them with error handling."""
        try:
            logger.info(f"Checking emails in folder '{folder}' (limit: {limit})")
            
            # Check health before operation
            health = self._check_health()
            if not health["retriever"]:
                logger.error("Cannot check emails: Email retriever is not healthy")
                return []
            
            # Get emails with retry logic
            retry_count = 0
            while retry_count < self.MAX_RETRIES:
                try:
                    emails = self.retriever.get_unread_emails(folder, limit)
                    self.performance_metrics["emails_checked"] += len(emails)
                    return emails
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error checking emails (attempt {retry_count}/{self.MAX_RETRIES}): {e}")
                    if retry_count >= self.MAX_RETRIES:
                        self.performance_metrics["errors"] += 1
                        break
                    time.sleep(2 ** retry_count)  # Exponential backoff
            
            return []
        except Exception as e:
            logger.error(f"Unexpected error checking emails: {e}")
            logger.debug(traceback.format_exc())
            self.performance_metrics["errors"] += 1
            return []
    
    def process_emails(self, auto_reply: bool = False) -> List[Dict[str, Any]]:
        """Process unread emails and optionally auto-reply with enhanced error handling."""
        try:
            logger.info(f"Processing emails (auto_reply: {auto_reply})")
            
            # Check health before operation
            health = self._check_health()
            if not health["retriever"] or not health["ai"] or (auto_reply and not health["sender"]):
                logger.error(f"Cannot process emails: Some components are not healthy: {json.dumps(health)}")
                return []
            
            emails = self.check_emails()
            
            if not emails:
                logger.info("No new emails to process")
                return []
            
            processed_emails = []
            for email_data in emails:
                try:
                    # Generate AI response
                    ai_response = self.ai.generate_response(email_data)
                    
                    if ai_response:
                        email_data["ai_response"] = ai_response
                        processed_emails.append(email_data)
                        self.performance_metrics["emails_processed"] += 1
                        
                        # Auto-reply if enabled
                        if auto_reply:
                            reply_success = self.sender.reply_to_email(email_data, ai_response)
                            email_data["reply_sent"] = reply_success
                            
                            if reply_success:
                                logger.info(f"Auto-replied to email: {email_data['subject']}")
                                self.performance_metrics["emails_replied"] += 1
                            else:
                                logger.error(f"Failed to auto-reply to email: {email_data['subject']}")
                    else:
                        logger.warning(f"Failed to generate AI response for email: {email_data['subject']}")
                except Exception as e:
                    logger.error(f"Error processing email {email_data.get('subject', 'Unknown')}: {e}")
                    logger.debug(traceback.format_exc())
                    self.performance_metrics["errors"] += 1
            
            return processed_emails
        except Exception as e:
            logger.error(f"Unexpected error processing emails: {e}")
            logger.debug(traceback.format_exc())
            self.performance_metrics["errors"] += 1
            return []
    
    def compose_new_email(self, to_address: str, subject: str, prompt: str, 
                         cc: Optional[Union[str, List[str]]] = None, 
                         bcc: Optional[Union[str, List[str]]] = None) -> bool:
        """Compose a new email with AI-generated content based on a prompt."""
        try:
            logger.info(f"Composing new email to {to_address} with subject: {subject}")
            
            # Check health before operation
            health = self._check_health()
            if not health["ai"] or not health["sender"]:
                logger.error(f"Cannot compose email: Some components are not healthy: {json.dumps(health)}")
                return False
            
            # Validate inputs
            if not to_address or "@" not in to_address:
                logger.error(f"Invalid 'to' address: {to_address}")
                return False
            
            if not subject:
                logger.warning("Email has no subject, using default")
                subject = "AI Generated Email"
            
            if not prompt:
                logger.error("No prompt provided for email content")
                return False
            
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
            result = self.sender.send_email(to_address, subject, ai_content, cc, bcc)
            
            if result:
                logger.info(f"Email composed and sent successfully to {to_address}")
                self.performance_metrics["emails_composed"] += 1
            else:
                logger.error(f"Failed to send composed email to {to_address}")
                self.performance_metrics["errors"] += 1
            
            return result
        except Exception as e:
            logger.error(f"Unexpected error composing email: {e}")
            logger.debug(traceback.format_exc())
            self.performance_metrics["errors"] += 1
            return False
    
    def draft_email(self, prompt: str) -> Optional[str]:
        """Draft an email without sending it."""
        try:
            logger.info(f"Drafting email with prompt: {prompt[:50]}...")
            
            # Check health before operation
            health = self._check_health()
            if not health["ai"]:
                logger.error(f"Cannot draft email: AI component is not healthy")
                return None
            
            if not prompt:
                logger.error("No prompt provided for email draft")
                return None
            
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
            
            logger.info("Email draft generated successfully")
            self.performance_metrics["emails_drafted"] += 1
            return ai_content
        except Exception as e:
            logger.error(f"Unexpected error drafting email: {e}")
            logger.debug(traceback.format_exc())
            self.performance_metrics["errors"] += 1
            return None
    
    def close_connections(self) -> None:
        """Close all server connections safely."""
        try:
            logger.info("Closing all connections")
            self.retriever.disconnect()
            self.sender.disconnect()
            logger.info("All connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all components."""
        # Calculate uptime
        uptime = time.time() - self.performance_metrics["start_time"]
        
        # Combine metrics from all components
        combined_metrics = {
            **self.performance_metrics,
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "retriever": self.retriever.get_performance_metrics(),
            "ai": self.ai.get_performance_metrics(),
            "sender": self.sender.get_performance_metrics(),
            "health": self.health_status
        }
        
        return combined_metrics
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in a human-readable format."""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        
        return " ".join(parts)


def validate_choice(choice: str, valid_choices: List[str]) -> bool:
    """Validate user input choice."""
    return choice in valid_choices


def main():
    """Main function to run the AI Email Agent with enhanced CLI."""
    # Setup argument parser for command-line options
    parser = argparse.ArgumentParser(description="AI Email Agent")
    parser.add_argument("--check", action="store_true", help="Check for unread emails")
    parser.add_argument("--process", action="store_true", help="Process emails without replying")
    parser.add_argument("--reply", action="store_true", help="Process emails and auto-reply")
    parser.add_argument("--compose", action="store_true", help="Compose a new email")
    parser.add_argument("--draft", action="store_true", help="Draft an email without sending")
    parser.add_argument("--metrics", action="store_true", help="Show performance metrics")
    parser.add_argument("--to", help="Recipient for compose mode")
    parser.add_argument("--subject", help="Subject for compose mode")
    parser.add_argument("--prompt", help="Prompt for compose or draft mode")
    parser.add_argument("--cc", help="CC recipients for compose mode (comma-separated)")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of emails to process")
    
    args = parser.parse_args()
    
    print("AI Email Agent")
    print("==============")
    
    # Initialize agent with error handling
    try:
        agent = AIEmailAgent()
    except Exception as e:
        print(f"Failed to initialize AI Email Agent: {e}")
        return 1
    
    # Process command-line arguments if provided
    if args.check or args.process or args.reply or args.compose or args.draft or args.metrics:
        try:
            if args.check:
                emails = agent.check_emails(limit=args.limit)
                if emails:
                    print(f"\nFound {len(emails)} unread emails:")
                    for i, email_data in enumerate(emails, 1):
                        print(f"{i}. From: {email_data['sender']}")
                        print(f"   Subject: {email_data['subject']}")
                        print(f"   Date: {email_data['date']}")
                        print()
                else:
                    print("\nNo unread emails found.")
            
            elif args.process:
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
            
            elif args.reply:
                processed = agent.process_emails(auto_reply=True)
                if processed:
                    print(f"\nAuto-replied to {len(processed)} emails.")
                else:
                    print("\nNo emails were auto-replied to.")
            
            elif args.compose:
                if not args.to or not args.prompt:
                    print("Error: --to and --prompt are required for compose mode")
                    return 1
                
                subject = args.subject or "AI Generated Email"
                cc = args.cc.split(",") if args.cc else None
                
                if agent.compose_new_email(args.to, subject, args.prompt, cc):
                    print("\nEmail composed and sent successfully!")
                else:
                    print("\nFailed to compose and send email.")
            
            elif args.draft:
                if not args.prompt:
                    print("Error: --prompt is required for draft mode")
                    return 1
                
                draft = agent.draft_email(args.prompt)
                
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
            
            elif args.metrics:
                metrics = agent.get_performance_metrics()
                print("\nPerformance Metrics:")
                print("===================")
                print(f"Uptime: {metrics['uptime_formatted']}")
                print(f"Emails checked: {metrics['emails_checked']}")
                print(f"Emails processed: {metrics['emails_processed']}")
                print(f"Emails replied: {metrics['emails_replied']}")
                print(f"Emails composed: {metrics['emails_composed']}")
                print(f"Emails drafted: {metrics['emails_drafted']}")
                print(f"Errors: {metrics['errors']}")
                print("\nHealth Status:")
                for component, status in metrics['health'].items():
                    print(f"  {component}: {'Healthy' if status else 'Unhealthy'}")
            
            agent.close_connections()
            return 0
            
        except Exception as e:
            print(f"Error: {e}")
            agent.close_connections()
            return 1
    
    # Implement interactive menu with input validation
    while True:
        print("\nOptions:")
        print("1. Check emails")
        print("2. Process emails (generate AI responses)")
        print("3. Process emails with auto-reply")
        print("4. Compose new email")
        print("5. Draft email")
        print("6. Show performance metrics")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if not validate_choice(choice, ["1", "2", "3", "4", "5", "6", "7"]):
            print("\nInvalid choice. Please enter a number between 1 and 7.")
            continue
        
        try:
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
                while not to_address or "@" not in to_address:
                    print("Invalid email address. Please enter a valid email.")
                    to_address = input("To: ")
                
                subject = input("Subject: ")
                
                prompt = input("What should the email be about? ")
                while not prompt:
                    print("Prompt cannot be empty. Please provide content for the email.")
                    prompt = input("What should the email be about? ")
                
                cc = input("CC (optional, separate multiple with commas): ")
                cc = [email.strip() for email in cc.split(",")] if cc else None
                
                if agent.compose_new_email(to_address, subject, prompt, cc):
                    print("\nEmail composed and sent successfully!")
                else:
                    print("\nFailed to compose and send email.")
            
            elif choice == "5":
                prompt = input("What should the email be about? ")
                while not prompt:
                    print("Prompt cannot be empty. Please provide content for the email.")
                    prompt = input("What should the email be about? ")
                
                draft = agent.draft_email(prompt)
                
                if draft:
                    print("\nEmail Draft:")
                    print("============")
                    print(draft)
                    print("============")
                    
                    save = input("\nSave draft to file? (y/n): ")
                    while save.lower() not in ["y", "n"]:
                        print("Invalid choice. Please enter 'y' or 'n'.")
                        save = input("Save draft to file? (y/n): ")
                    
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
                metrics = agent.get_performance_metrics()
                print("\nPerformance Metrics:")
                print("===================")
                print(f"Uptime: {metrics['uptime_formatted']}")
                print(f"Emails checked: {metrics['emails_checked']}")
                print(f"Emails processed: {metrics['emails_processed']}")
                print(f"Emails replied: {metrics['emails_replied']}")
                print(f"Emails composed: {metrics['emails_composed']}")
                print(f"Emails drafted: {metrics['emails_drafted']}")
                print(f"Errors: {metrics['errors']}")
                print("\nHealth Status:")
                for component, status in metrics['health'].items():
                    print(f"  {component}: {'Healthy' if status else 'Unhealthy'}")
            
            elif choice == "7":
                agent.close_connections()
                print("\nThank you for using AI Email Agent. Goodbye!")
                break
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            logger.error(f"Error in main loop: {e}")
            logger.debug(traceback.format_exc())
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)