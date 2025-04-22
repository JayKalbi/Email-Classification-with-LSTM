
import os
import pickle
import base64
import time
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText

# Constants
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.send'
]

MAX_LENGTH = 250
LABELS = [
    'Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics',
    'Related to Other People', 'Places', 'Legal', 'Business',
    '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA'
]

# Custom responses per label
RESPONSES = {
    'Weather/Natural': "Thank you for sharing weather-related information.",
    'Sent Mail': "This appears to be a copy of a sent mail.",
    'Random/NA': "We received your message. Thank you!",
    'Financial/Logistics': "Your message regarding financial or logistics matters has been noted.",
    'Related to Other People': "Thanks for your message regarding other individuals.",
    'Places': "Your message concerning specific places has been acknowledged.",
    'Legal': "We’ve received your legal-related inquiry and will respond accordingly.",
    'Business': "Thank you for reaching out with a business matter.",
    '2-Letter/Random': "Your message has been logged for review.",
    'Other Firms': "We acknowledge your reference to external firms.",
    'HR/Recruiting/MBA': "Thank you for your interest in HR or recruiting topics."
}

# Authentication
def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

# Clean email text
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower().strip()

# Load model and tokenizer
model = tf.keras.models.load_model('Models/LSTM_no1n2.h5')
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Classify email
def classify_email(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH)
    pred = model.predict(padded)
    label_idx = np.argmax(pred)
    return LABELS[label_idx], pred[0][label_idx]

# Send auto-response
def send_response(service, msg_id, to, label):
    message_text = RESPONSES.get(label, "Thank you for your message.")
    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = f"RE: Auto Response Regarding {label}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    print(f"Sent response to {to} for label: {label}")

# Main handler
def process_emails():
    service = authenticate_gmail()
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread", maxResults=10).execute()
    messages = results.get('messages', [])

    if not messages:
        print("No new emails found.")
        return

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), None)

        if not sender:
            continue

        parts = msg_data['payload'].get('parts', [])
        body = ''
        for part in parts:
            if part['mimeType'] == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                break

        if not body:
            continue

        label, confidence = classify_email(body)
        print(f"Subject: {subject}\nSender: {sender}\nPredicted Label: {label} ({confidence:.2f})")

        # Create label if not exists
        label_list = service.users().labels().list(userId='me').execute()
        existing_labels = [l['name'] for l in label_list['labels']]
        if label not in existing_labels:
            service.users().labels().create(userId='me', body={'name': label}).execute()

        # Apply label
        label_id = next((l['id'] for l in label_list['labels'] if l['name'] == label), None)
        if label_id:
            service.users().messages().modify(userId='me', id=msg['id'], body={
                'addLabelIds': [label_id],
                'removeLabelIds': ['INBOX']
            }).execute()

        # Send response
        send_response(service, msg['id'], sender, label)

        # Mark as read
        service.users().messages().modify(userId='me', id=msg['id'], body={
            'removeLabelIds': ['UNREAD']
        }).execute()

# If run as script
if __name__ == "__main__":
    process_emails()
