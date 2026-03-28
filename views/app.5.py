# ==============================
# Project Structure
# ==============================
# views/
#   ├── app.py
# utils/
#   ├── loaders.py
#   ├── preprocessing.py
#   ├── prediction.py
#   ├── similarity.py
#   ├── rag.py

# ==============================
# app.py
# ==============================
import streamlit as st
from utils.loaders import *
from utils.preprocessing import clean_text
from utils.prediction import predict_all
from utils.similarity import evaluate_similarity
from utils.rag import run_rag

import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance NLP App", layout="wide")

# Load resources
with st.spinner("Loading models..."):
    df = load_dataframe()
    mark_model = load_tfidf_mark_model()
    sa_model = load_tfidf_sa_model()
    bert_pipe = load_bert_pipeline()
    distil_pipe = load_distilbert_pipeline()
    subject_model, label_embs, labels = load_subject_model()
    data_embeddings = load_data_emeddings()

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

        subject_scores = evaluate_similarity(clean, subject_model, label_embs, labels)

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
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    margin=dict(l=10, r=10, t=30, b=10),
)

PALETTE_SENT = {
    "Positif": "#48c78e",
    "Neutre": "#ffbd59",
    "Négatif": "#ff6b6b",
}

PALETTE_BLUE = ["#1a3a5c", "#244e7c", "#2f629c", "#3b82f6", "#63b3ed"]

with tab2:
    note_col = "note"

    insurer = st.selectbox("Filter by insurer", ["All"] + list(df['assureur'].unique()))
    filtered_df = df if insurer == "All" else df[df['assureur'] == insurer]

    st.subheader("Moyenne par Assureur")
    st.write(filtered_df.groupby('assureur')['note'].mean())

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.subheader("Distribution des notes")
        if note_col in filtered_df.columns:
            note_counts = filtered_df[note_col].value_counts().sort_index()
            fig_notes = go.Figure(go.Bar(
                x=[str(int(n)) for n in note_counts.index],
                y=note_counts.values,
                marker=dict(color=note_counts.values,
                            colorscale=[[0, "#1a3a5c"], [1, "#63b3ed"]]),
                text=note_counts.values,
                textposition="outside",
            ))
            fig_notes.update_layout(**PLOTLY_LAYOUT, height=320,
                                    xaxis_title="Note (1 à 5)", yaxis_title="Nombre d'avis")
            st.plotly_chart(fig_notes, width='stretch')

    with col_b:
        st.subheader("Répartition des sentiments")
        if "sentiment" in filtered_df.columns:
            sent_counts = filtered_df["sentiment"].value_counts()
            fig_sent = go.Figure(go.Pie(
                labels=sent_counts.index.tolist(),
                values=sent_counts.values.tolist(),
                hole=0.55,
                marker=dict(colors=[PALETTE_SENT.get(s, "#888") for s in sent_counts.index]),
            ))
            fig_sent.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_sent, width='stretch')

    st.subheader("Top 20 mots les plus fréquents")
    if "text_cleaned" in filtered_df.columns:
        all_words = " ".join(filtered_df["text_cleaned"].astype(str)).split()
        word_freq = Counter(w for w in all_words if len(w) > 2)
        common = word_freq.most_common(20)
        words_list = [w for w, _ in common]
        counts_list = [c for _, c in common]

        fig_words = go.Figure(go.Bar(
            x=counts_list[::-1],
            y=words_list[::-1],
            orientation="h",
            marker=dict(color=counts_list[::-1],
                        colorscale=[[0, "#1a3a5c"], [1, "#63b3ed"]]),
            text=counts_list[::-1],
            textposition="outside",
        ))
        fig_words.update_layout(**PLOTLY_LAYOUT, height=480,
                                xaxis_title="Fréquence")
        st.plotly_chart(fig_words, width='stretch')

    if "subject" in filtered_df.columns and filtered_df["subject"].nunique() > 1:
        st.subheader("Détection de sujets")
        col_s1, col_s2 = st.columns([1, 1], gap="large")

        with col_s1:
            subj_counts = filtered_df["subject"].value_counts()
            fig_subj = go.Figure(go.Bar(
                x=subj_counts.index.tolist(),
                y=subj_counts.values.tolist(),
                marker=dict(color=PALETTE_BLUE[:len(subj_counts)]),
                text=subj_counts.values.tolist(),
                textposition="outside",
            ))
            fig_subj.update_layout(**PLOTLY_LAYOUT, height=320,
                                   xaxis_title="Catégorie", yaxis_title="Nombre d'avis")
            st.plotly_chart(fig_subj, width='stretch')

        with col_s2:
            if note_col in filtered_df.columns:
                subj_note = filtered_df.groupby("subject")[note_col].mean().sort_values()
                fig_subj_note = go.Figure(go.Bar(
                    x=subj_note.values,
                    y=subj_note.index.tolist(),
                    orientation="h",
                    marker=dict(color=subj_note.values,
                                colorscale=[[0, "#ff6b6b"], [1, "#48c78e"]]),
                    text=[f"{v:.2f}" for v in subj_note.values],
                    textposition="outside",
                ))
                fig_subj_note.update_layout(**PLOTLY_LAYOUT, height=320,
                                            xaxis_title="Note moyenne")
                st.plotly_chart(fig_subj_note, width='stretch')

    st.subheader("Nuage de mots")

    col_wc1, col_wc2 = st.columns([2, 1])

    with col_wc2:
        colormap_choice = st.selectbox("Palette", ["plasma", "viridis", "magma", "cool", "Blues"])
        max_words_wc = st.slider("Nombre de mots max", 30, 200, 100)
        filter_sentiment_wc = st.selectbox("Filtrer par sentiment",
                                           ["Tous", "Positif", "Neutre", "Négatif"])

    with col_wc1:
        try:
            from wordcloud import WordCloud
            subset = filtered_df
            if filter_sentiment_wc != "Tous" and "sentiment" in filtered_df.columns:
                subset = filtered_df[filtered_df["sentiment"] == filter_sentiment_wc]

            full_text = " ".join(subset["text_cleaned"].astype(str))

            if len(full_text.strip()) > 10:
                wc = WordCloud(width=900, height=420,
                               background_color=None,
                               mode="RGBA",
                               colormap=colormap_choice,
                               max_words=max_words_wc).generate(full_text)

                fig_wc, ax_wc = plt.subplots(figsize=(9, 4.2))
                ax_wc.imshow(wc)
                ax_wc.axis("off")
                st.pyplot(fig_wc)

        except ImportError:
            st.warning("Installe wordcloud : pip install wordcloud")

# ==============================
# TAB 3 - RAG
# ==============================
with tab3:
    st.subheader("Search")
    query = st.text_input("Search reviews")
    if query:
        embs = data_embeddings[filtered_df.index]
        idx2scores = evaluate_similarity(query, subject_model, embs, range(len(embs)))
        results_idxs = [idx for idx, score in idx2scores if score >= 0.4]
        results_df = filtered_df.iloc[results_idxs]
        cols_to_show = [
            c
            for c in ["text_cleaned", "note", "sentiment", "subject"]
            if c in results_df.columns
        ]
        st.dataframe(results_df.head(20).loc[:,cols_to_show])

    st.subheader("Q/A")
    question = st.text_area("Ask a question")
    if st.button("Run"):
        try:
            answer = run_rag(question, filtered_df)
            st.write(answer)
        except Exception as e:
            st.error("Ollama not running")
