import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU, use CPU

import base64
import pickle
import numpy as np
from email import message_from_bytes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Terminal colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Email classification labels
labels = ['Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics',
          'Related to Other People', 'Places', 'Legal', 'Business',
          '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA']

# Load model and tokenizer
model = load_model('Models/LSTM_no1n2.h5')
with open('Models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_length = 250
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def get_last_15_email_bodies(service):
    email_bodies = []
    message_ids = []
    next_page_token = None

    # Continue fetching until we get at least 15 valid email bodies
    while len(email_bodies) < 15:
        results = service.users().messages().list(
            userId='me',
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        messages = results.get('messages', [])
        next_page_token = results.get('nextPageToken')

        if not messages:
            break  # No more messages available

        for msg in messages:
            try:
                msg_data = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
                raw_msg = base64.urlsafe_b64decode(msg_data['raw'].encode('ASCII'))
                mime_msg = message_from_bytes(raw_msg)

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
                    body = mime_msg.get_payload(decode=True).decode()

                if body.strip():
                    email_bodies.append(body.strip())

                if len(email_bodies) == 15:
                    return email_bodies

            except Exception as e:
                continue  # Skip malformed emails

        if not next_page_token:
            break  # No more pages to fetch

    return email_bodies


def classify_emails(email_texts):
    print("\n" + "="*80)
    print(f"{bcolors.BOLD}{bcolors.OKCYAN}Classifying the Last 15 Emails...{bcolors.ENDC}")
    print("="*80)

    for idx, text in enumerate(email_texts, 1):
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_length)
        prediction = model.predict(padded, verbose=0)
        predicted_label = labels[np.argmax(prediction)]

        print(f"\n{bcolors.BOLD}Email #{idx}:{bcolors.ENDC}")
        print(f"{bcolors.OKGREEN}Predicted Label:{bcolors.ENDC} {bcolors.BOLD}{predicted_label}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Preview:{bcolors.ENDC} {text[:200]}...")

    print("\n" + "="*80)
    print(f"{bcolors.OKCYAN}Classification Complete.{bcolors.ENDC}")
    print("="*80)

def main():
    service = authenticate_gmail()
    email_texts = get_last_15_email_bodies(service)
    classify_emails(email_texts)

if __name__ == '__main__':
    main()
