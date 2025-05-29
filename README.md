# AI Email Agent

A Python script for an AI agent that can respond to emails, check emails, and draft new emails.

## Features

- **Email Retrieval**: Connect to email servers via IMAP to fetch and parse emails
- **AI Response Generation**: Generate intelligent responses to emails using OpenAI's GPT API
- **Email Composition**: Create and send new emails with AI-generated content
- **Email Drafting**: Draft emails without sending them
- **Auto-Reply**: Automatically respond to incoming emails

## Requirements

- Python 3.6+
- OpenAI API key for AI response generation
- Email account with IMAP and SMTP access

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
6. **Exit**: Close all connections and exit the program

## Configuration

On first run, you'll be prompted to enter your email configuration:
- Email address
- Password
- IMAP server (e.g., imap.gmail.com)
- SMTP server (e.g., smtp.gmail.com)
- IMAP port (default: 993)
- SMTP port (default: 587)

This information will be saved to `email_config.ini` for future use.

## Gmail Setup

If you're using Gmail, you'll need to:
1. Enable "Less secure app access" or
2. Use an "App Password" if you have 2-factor authentication enabled

## Security Note

Your email password is stored in the configuration file. For better security:
- Ensure the file has appropriate permissions
- Consider using environment variables instead
- Use an app-specific password rather than your main account password

## License

MIT

## Disclaimer

This tool is for educational and personal use only. Be responsible when using automated email systems.