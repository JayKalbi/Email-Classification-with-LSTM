import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="📊 Model Performance", layout="wide")
st.title("📊 Model Results & Performance")

# === CONFIGURATION ===
MODEL_PATH = "Models/LSTM_no1n2.h5"
TOKENIZER_PATH = "Models/tokenizer.pickle"
LABELS = ['Weather/Natural', 'Sent Mail', 'Random/NA', 'Financial/Logistics',
          'Related to Other People', 'Places', 'Legal', 'Business',
          '2-Letter/Random', 'Other Firms', 'HR/Recruiting/MBA']
MAX_LENGTH = 250

# === LOAD MODEL & TOKENIZER ===
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def load_history():
    if os.path.exists("Models/history.pickle"):
        with open("Models/history.pickle", "rb") as f:
            return pickle.load(f)
    return None

model, tokenizer = load_model_and_tokenizer()
history = load_history()

# === SAMPLE TEST DATA (for now) ===
y_test = np.random.randint(0, len(LABELS), 100)
X_sample = ["This is a test email about business and logistics."] * 100
sequences = tokenizer.texts_to_sequences(X_sample)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH)
y_pred = model.predict(padded)
y_pred_labels = np.argmax(y_pred, axis=1)

# === METRICS DISPLAY ===
st.subheader("📌 Classification Report")
report = classification_report(y_test, y_pred_labels, target_names=LABELS, zero_division=0)
st.text(report)

acc = accuracy_score(y_test, y_pred_labels)
st.write(f"**✅ Accuracy:** {acc:.2f}")

# === CONFUSION MATRIX ===
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_labels)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
st.pyplot(fig)

# === TRAINING CURVES ===
if history:
    st.subheader("📈 Training Accuracy and Loss")
    fig1, ax1 = plt.subplots()
    ax1.plot(history['accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(history['loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    st.pyplot(fig2)
else:
    st.info("📁 No training history file found. Add 'Models/history.pickle' to enable training plots.")
