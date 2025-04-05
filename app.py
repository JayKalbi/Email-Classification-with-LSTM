import streamlit as st
import base64
import os
import pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from bs4 import BeautifulSoup
import re

# === CONFIGURATION ===
MODEL_PATH = "Models/LSTM_no1n2.h5"
TOKENIZER_PATH = "Models/tokenizer.pickle"
LABELS = ['Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics',
          'Related to Other People', 'Places', 'Legal', 'Business',
          '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA']
MAX_LENGTH = 250
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


# === HELPER FUNCTIONS ===
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # remove excessive whitespace
    return text.strip()

def fetch_latest_emails(n=10):
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    service = build('gmail', 'v1', credentials=creds)

    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=n).execute()
    messages = results.get('messages', [])

    email_texts = []

    for msg in messages:
        msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        payload = msg_detail.get('payload', {})
        parts = payload.get('parts', [])
        data = ""

        for part in parts:
            if part.get('mimeType') == 'text/plain':
                data = part['body'].get('data')
                break
            elif part.get('mimeType') == 'text/html':
                data = part['body'].get('data')
                break

        if data:
            decoded_bytes = base64.urlsafe_b64decode(data.encode('UTF-8'))
            soup = BeautifulSoup(decoded_bytes, "html.parser")
            text = soup.get_text()
            email_texts.append(clean_text(text))

    return email_texts


# === STREAMLIT APP ===
st.set_page_config(page_title="📧 Email Classification Dashboard", layout="wide")
st.title("📧 Gmail Email Classification using LSTM")
st.write("This app fetches your latest Gmail emails and classifies them using a trained LSTM model.")

try:
    model, tokenizer = load_model_and_tokenizer()
    email_texts = fetch_latest_emails(n=10)

    if not email_texts:
        st.warning("No emails fetched.")
        st.stop()

    st.success(f"✅ Fetched {len(email_texts)} latest emails!")

    # Classification
    sequences = tokenizer.texts_to_sequences(email_texts)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH)
    predictions = model.predict(padded, verbose=0)
    predicted_labels = [LABELS[np.argmax(pred)] for pred in predictions]
    confidences = [round(float(np.max(p)) * 100, 2) for p in predictions]

    # Display raw results
    st.subheader("📬 Classified Emails")
    for i, text in enumerate(email_texts):
        st.markdown(f"**✉️ Email #{i+1}**")
        st.write(text[:500] + "..." if len(text) > 500 else text)
        st.write(f"🔖 **Predicted Label:** {predicted_labels[i]} — 💯 **Confidence:** {confidences[i]}%")
        st.markdown("---")

    # === VISUALIZATIONS ===

    # DataFrame for charts
    df = pd.DataFrame({
        "Email": [t[:50] + "..." for t in email_texts],
        "Predicted Label": predicted_labels,
        "Confidence (%)": confidences
    })

    st.subheader("📊 Label Distribution")
    label_counts = df["Predicted Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    fig_bar = px.bar(label_counts, x="Label", y="Count", color="Label")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 Confidence Scores")
    fig_conf = px.bar(df, x="Email", y="Confidence (%)", color="Predicted Label", height=400)
    st.plotly_chart(fig_conf, use_container_width=True)

    st.subheader("☁️ Word Cloud by Predicted Label")
    selected_label = st.selectbox("Choose a label", options=list(set(predicted_labels)))
    combined_text = " ".join([email_texts[i] for i, label in enumerate(predicted_labels) if label == selected_label])
    if combined_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)

    st.success("✅ Classification and Visualization Complete!")

except Exception as e:
    st.error("An error occurred while fetching or classifying your emails.")
    st.exception(e)
