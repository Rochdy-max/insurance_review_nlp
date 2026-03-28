# ==============================
# utils/loaders.py
# ==============================
import pandas as pd
import joblib
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer

@st.cache_data
def load_dataframe():
    return pd.read_csv("data/final_data.csv")

@st.cache_resource
def load_tfidf_mark_model():
    return joblib.load("models/tf_idf_clf_mark.joblib")

@st.cache_resource
def load_tfidf_sa_model():
    return joblib.load("models/tf_idf_clf_sa.joblib")

@st.cache_resource
def load_bert_pipeline():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def load_distilbert_pipeline():
    return pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@st.cache_resource
def load_subject_model():
    labels = ["Pricing", "Coverage", "Enrollment", "Customer Service", "Claims Processing", "Cancellation"]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    label_embs = model.encode(labels)
    return model, label_embs, labels

@st.cache_resource
def load_data_emeddings():
    return pd.read_pickle("data/text_embeddings.pkl")
