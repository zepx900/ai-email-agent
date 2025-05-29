# Enhanced AI Email Agent

A Python script for an AI agent that can respond to emails, check emails, and draft new emails with improved correctness, functionality, performance, security, and fault tolerance.

## Features

- **Email Retrieval**: Connect to email servers via IMAP to fetch and parse emails
- **AI Response Generation**: Generate intelligent responses to emails using OpenAI's GPT API
- **Email Composition**: Create and send new emails with AI-generated content
- **Email Drafting**: Draft emails without sending them
- **Auto-Reply**: Automatically respond to incoming emails
- **Enhanced Security**: Encrypted credential storage and secure connections
- **Improved Performance**: Caching, connection pooling, and performance metrics
- **Fault Tolerance**: Comprehensive error handling, retry logic, and health checks
- **Command-line Interface**: Both interactive and argument-based modes

## Requirements

- Python 3.6+
- OpenAI API key for AI response generation
- Email account with IMAP and SMTP access
- Required packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/zepx900/ai-email-agent.git
   cd ai-email-agent
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Option 1: Set it as an environment variable:
     ```
     export OPENAI_API_KEY=your_api_key_here
     ```
   - Option 2: Enter it when prompted by the script

## Usage

### Interactive Mode

Run the script:
```
python ai_email_agent.py
```

The script will guide you through the following options:

1. **Check emails**: Retrieve and display unread emails
2. **Process emails**: Generate AI responses for unread emails without sending replies
3. **Process emails with auto-reply**: Generate AI responses and automatically send replies
4. **Compose new email**: Create and send a new email with AI-generated content
5. **Draft email**: Generate an email draft without sending it
6. **Show performance metrics**: Display statistics about the agent's performance
7. **Exit**: Close all connections and exit the program

### Command-line Arguments

You can also use command-line arguments for non-interactive usage:

```
python ai_email_agent.py --check                      # Check for unread emails
python ai_email_agent.py --process                    # Process emails without replying
python ai_email_agent.py --reply                      # Process emails and auto-reply
python ai_email_agent.py --compose --to user@example.com --prompt "Schedule a meeting" # Compose email
python ai_email_agent.py --draft --prompt "Proposal for new project"  # Draft an email
python ai_email_agent.py --metrics                    # Show performance metrics
```

Additional options:
- `--to`: Recipient email address for compose mode
- `--subject`: Email subject for compose mode
- `--prompt`: Content prompt for compose or draft mode
- `--cc`: CC recipients for compose mode (comma-separated)
- `--limit`: Maximum number of emails to process

## Configuration

On first run, you'll be prompted to enter your email configuration:
- Email address
- Password
- IMAP server (e.g., imap.gmail.com)
- SMTP server (e.g., smtp.gmail.com)
- IMAP port (default: 993)
- SMTP port (default: 587)

This information will be securely stored in `email_config.ini` for future use.

## Security Features

- Credentials are encrypted in the configuration file
- Secure file permissions for the configuration file
- TLS encryption for email connections
- Input validation to prevent injection attacks
- Option to use environment variables for sensitive data

## Performance Optimizations

- Connection pooling and reuse
- Caching for repeated operations
- Exponential backoff for retries
- Performance metrics tracking

## Fault Tolerance

- Comprehensive error handling
- Automatic reconnection on failure
- Fallback mechanisms for AI generation
- Health checks for all components
- Detailed logging for troubleshooting

## Gmail Setup

If you're using Gmail, you'll need to:
1. Enable "Less secure app access" or
2. Use an "App Password" if you have 2-factor authentication enabled

## License

MIT

## Disclaimer

This tool is for educational and personal use only. Be responsible when using automated email systems.