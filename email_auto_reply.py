import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU

import base64
import pickle
import numpy as np
from email import message_from_bytes
from email.mime.text import MIMEText

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Gmail API scope with read + send access
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Load model and tokenizer
model = load_model('Models/LSTM_no1n2.h5')
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Labels used by model
labels = ['Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics',
          'Related to Other People', 'Places', 'Legal', 'Business',
          '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def classify_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=250)
    prediction = model.predict(padded, verbose=0)
    return labels[np.argmax(prediction)]

def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

def send_message(service, user_id, message):
    sent_msg = service.users().messages().send(userId=user_id, body=message).execute()
    print(f"✅ Auto-reply sent! Message ID: {sent_msg['id']}")

def process_emails(service):
    results = service.users().messages().list(userId='me', q="is:unread -from:me", maxResults=10).execute()
    messages = results.get('messages', [])

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        raw_msg = base64.urlsafe_b64decode(msg_data['raw'].encode('ASCII'))
        mime_msg = message_from_bytes(raw_msg)

        # Extract sender and subject
        headers = msg_data['payload']['headers']
        sender = next((h['value'] for h in headers if h['name'] == 'From'), None)
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "(No Subject)")

        # Extract body
        body = ""
        if mime_msg.is_multipart():
            for part in mime_msg.walk():
                if part.get_content_type() == 'text/plain':
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        continue
        else:
            try:
                body = mime_msg.get_payload(decode=True).decode()
            except:
                continue

        if not body:
            continue

        # Classify
        label = classify_text(body)
        print(f"\n✉️ From: {sender}\nSubject: {subject}\n→ Predicted Label: {label}")

        # Auto-reply
        reply_text = f"""Hi,

Thanks for your email. Based on our AI system, we classified your message under the category: **{label}**.

We'll get back to you shortly if necessary.

Regards,
AutoClassifier Bot
"""
        reply_msg = create_message("me", sender, f"Re: {subject}", reply_text)
        send_message(service, "me", reply_msg)

        # Mark message as read
        service.users().messages().modify(userId='me', id=msg['id'],
            body={'removeLabelIds': ['UNREAD']}).execute()

def main():
    service = authenticate_gmail()
    process_emails(service)

if __name__ == '__main__':
    main()
