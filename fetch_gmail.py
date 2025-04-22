import os.path
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying these SCOPES, delete the token.json file.
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]


def clean_subject(subject):
    """Cleans the subject line for display"""
    return re.sub(r'[\r\n\t]', '', subject)

def main():
    creds = None

    # token.json stores access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no (valid) credentials available, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for next time
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Connect to Gmail API
    service = build('gmail', 'v1', credentials=creds)

    # Call Gmail API to fetch inbox messages
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
    messages = results.get('messages', [])

    if not messages:
        print("No messages found.")
    else:
        print("\n📥 Latest Emails:")
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['Subject']).execute()
            headers = msg_data.get('payload', {}).get('headers', [])
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "(No Subject)")
            print("•", clean_subject(subject))

if __name__ == '__main__':
    main()
