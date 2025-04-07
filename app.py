import streamlit as st

st.set_page_config(page_title="📧 AI Email Classifier", layout="wide")
st.title("📧 Smart Email Classifier Dashboard")

st.write("""
Welcome to the AI-powered Gmail email classification dashboard.  
Use the sidebar to navigate between:
- 📩 **Classify Emails**: Fetch, analyze, and label your Gmail emails using an LSTM model.
- 📊 **Model Performance**: Visualize model accuracy, confusion matrix, and training progress.
""")
