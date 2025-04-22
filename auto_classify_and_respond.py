from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from base64 import urlsafe_b64decode
from email.mime.text import MIMEText
import base64
import re

# Load model and tokenizer
model = load_model("Models/LSTM_n01n2.h5")
with open("tokenizerALL.pickle", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 250
LABELS = ['Legal', 'Financial', 'HR/Recruiting', 'Trading', 'Logistics/Operations',
          'IT/Technical Support', 'Management/Strategy', 'Marketing',
          'Customer Service', 'Personal', 'Spam/Promotions']

# Setup Gmail API
creds = Credentials.from_authorized_user_file("token.json", ['https://www.googleapis.com/auth/gmail.modify'])
service = build('gmail', 'v1', credentials=creds)

# Fetch last 10 emails
def get_last_10_emails():
    results = service.users().messages().list(userId='me', maxResults=10, labelIds=['INBOX']).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "")
        body = ""
        if 'data' in msg_data['payload']['body']:
            body = urlsafe_b64decode(msg_data['payload']['body']['data']).decode('utf-8', errors='ignore')
        emails.append({'id': msg['id'], 'sender': sender, 'subject': subject, 'body': body})
    return emails

# Preprocess and predict
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    return padded

def predict_label(text):
    padded = preprocess(text)
    pred = model.predict(padded)
    return LABELS[pred.argmax()]

# Label creation and email movement
def create_label_if_not_exists(label_name):
    labels = service.users().labels().list(userId='me').execute().get('labels', [])
    if not any(label['name'] == label_name for label in labels):
        label = {'name': label_name, 'labelListVisibility': 'labelShow', 'messageListVisibility': 'show'}
        service.users().labels().create(userId='me', body=label).execute()

def move_email_to_label(msg_id, label_name):
    create_label_if_not_exists(label_name)
    label_id = service.users().labels().list(userId='me').execute()
    label_id = next(label['id'] for label in label_id['labels'] if label['name'] == label_name)
    service.users().messages().modify(userId='me', id=msg_id, body={'addLabelIds': [label_id]}).execute()

# Auto response
def send_auto_reply(to, subject, label):
    message_text = f"Hello,\n\nThank you for your message regarding **{label}**. We've received your email and will get back to you shortly.\n\nBest,\nAutomated Classifier"
    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = f"RE: {subject}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {'raw': raw}
    service.users().messages().send(userId='me', body=body).execute()

# Execute
if __name__ == "__main__":
    emails = get_last_10_emails()
    for email in emails:
        label = predict_label(email['body'])
        print(f"Email from {email['sender']} predicted as {label}")
        move_email_to_label(email['id'], label)
        send_auto_reply(email['sender'], email['subject'], label)