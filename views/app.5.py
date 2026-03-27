# ==============================
# Project Structure
# ==============================
# views/
#   ├── app.py
# utils/
#   ├── loaders.py
#   ├── preprocessing.py
#   ├── prediction.py
#   ├── subject.py
#   ├── rag.py

# ==============================
# app.py
# ==============================
import streamlit as st
from utils.loaders import *
from utils.preprocessing import clean_text
from utils.prediction import predict_all
from utils.subject import predict_subject
from utils.rag import run_rag

st.set_page_config(page_title="Insurance NLP App", layout="wide")

# Load resources
with st.spinner("Loading models..."):
    df = load_dataframe()
    mark_model = load_tfidf_mark_model()
    sa_model = load_tfidf_sa_model()
    bert_pipe = load_bert_pipeline()
    distil_pipe = load_distilbert_pipeline()
    subject_model, label_embs, labels = load_subject_model()

st.success("Models loaded")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Analysis", "Q&A"])

# ==============================
# TAB 1 - Prediction
# ==============================
with tab1:
    text = st.text_area("Enter a review")
    if st.button("Predict") and text:
        clean = clean_text(text)

        mark_pred, mark_conf, bert_pred, bert_conf = predict_all(
            clean, mark_model, bert_pipe
        )

        sa_pred, sa_conf, distil_pred, distil_conf = predict_all(
            clean, sa_model, distil_pipe, task="sentiment"
        )

        subject_scores = predict_subject(clean, subject_model, label_embs, labels)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Rating")
            st.write(f"TF-IDF: {mark_pred} ({mark_conf:.2f})")
            st.write(f"BERT: {bert_pred} ({bert_conf:.2f})")

        with col2:
            st.subheader("Sentiment")
            st.write(f"TF-IDF: {sa_pred} ({sa_conf:.2f})")
            st.write(f"DistilBERT: {distil_pred} ({distil_conf:.2f})")

        st.subheader("Subject")
        for label, score in subject_scores:
            st.progress(float(score), text=f"{label}: {score:.2f}")

# ==============================
# TAB 2 - Analysis
# ==============================
with tab2:
    insurer = st.selectbox("Filter by insurer", ["All"] + list(df['assureur'].unique()))

    filtered_df = df if insurer == "All" else df[df['assureur'] == insurer]

    st.subheader("Metrics")
    st.write(filtered_df.groupby('assureur')['note'].mean())

    st.subheader("Search")
    query = st.text_input("Search reviews")
    if query:
        results = filtered_df[filtered_df['text_cleaned'].str.contains(query, case=False)]
        st.dataframe(results.head(20))

# ==============================
# TAB 3 - RAG
# ==============================
with tab3:
    question = st.text_area("Ask a question")
    if st.button("Run"):
        try:
            answer = run_rag(question, df)
            st.write(answer)
        except Exception as e:
            st.error("Ollama not running")
