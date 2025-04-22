import streamlit as st
import os
import subprocess

st.set_page_config(page_title="📧 AI Email Classifier", layout="wide")
st.title("📧 Smart Email Classifier Dashboard")

st.write("""
Welcome to the AI-powered Gmail email classification dashboard.  
Use the sidebar to navigate between:
- 📩 **Classify Emails**: Fetch, analyze, and label your Gmail emails using an LSTM model.
- 📊 **Model Performance**: Visualize model accuracy, confusion matrix, and training progress.
""")

st.markdown("## 📧 Gmail Auto Classification + Auto Response")



if st.button("📩 Auto-Label + Respond to Last 10 Emails"):
    with st.spinner("Running Gmail Classifier..."):
        subprocess.run(["python", "gmail_label_classifier_responder.py"])
    st.success("Emails processed and responses sent!")