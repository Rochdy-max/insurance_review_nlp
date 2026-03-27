"""
Insurance Review NLP - Application Streamlit
Analyse de sentiments, prédiction de notes et détection de sujets
sur des avis clients dans le secteur de l'assurance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import zipfile
import glob
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance NLP Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Fond principal */
.stApp {
    background: #0f0f14;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #16161e !important;
    border-right: 1px solid #2a2a38;
}

/* Titres */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2d4a7a;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #e8e4dc;
    margin: 0;
    line-height: 1.1;
}
.hero-subtitle {
    color: #63b3ed;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    margin-top: 8px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #1a1a26;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #63b3ed; }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #63b3ed;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e4dc;
    border-left: 3px solid #63b3ed;
    padding-left: 16px;
    margin: 32px 0 16px;
}

/* Sentiment badges */
.badge-pos {
    background: rgba(72,199,142,0.15);
    color: #48c78e;
    border: 1px solid rgba(72,199,142,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}
.badge-neu {
    background: rgba(255,189,89,0.15);
    color: #ffbd59;
    border: 1px solid rgba(255,189,89,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}
.badge-neg {
    background: rgba(255,100,100,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,100,100,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
}

/* Prediction result box */
.prediction-box {
    background: linear-gradient(135deg, #1a1a26, #1e1e2e);
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
}
.star-display {
    font-size: 3rem;
    line-height: 1;
}
.stars-score {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffd700;
    margin-top: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a26;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #888;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #2a2a3a !important;
    color: #63b3ed !important;
}

/* Plotly charts dark background override */
.js-plotly-plot .plotly { background: transparent !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1a26;
    border: 2px dashed #2a2a3a;
    border-radius: 12px;
    padding: 16px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f0f14; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK THEME ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#e8e4dc", size=12),
    title_font=dict(family="Syne, sans-serif", size=16, color="#e8e4dc"),
    # xaxis=dict(gridcolor="#2a2a3a", linecolor="#2a2a3a", tickcolor="#555"),
    # yaxis=dict(gridcolor="#2a2a3a", linecolor="#2a2a3a", tickcolor="#555"),
    margin=dict(l=20, r=20, t=40, b=20),
)

PALETTE_BLUE = ["#1a3a5c", "#1e4d7b", "#2563a8", "#3b82f6", "#63b3ed", "#93c5fd", "#bfdbfe"]
PALETTE_SENT = {"Positif": "#48c78e", "Neutre": "#ffbd59", "Négatif": "#ff6b6b"}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def note_to_sentiment(note):
    note = int(note)
    if note <= 2: return "Négatif"
    elif note == 3: return "Neutre"
    return "Positif"

def stars(note):
    n = int(note)
    return "⭐" * n + "☆" * (5 - n)

@st.cache_resource
def load_nlp_tools():
    import spacy
    from spellchecker import SpellChecker
    spell = SpellChecker(language="en")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return spell, nlp

@st.cache_resource
def load_tfidf_model():
    """Load or train a TF-IDF model. Returns None if no training data."""
    import joblib
    model_path = "models/tf_idf_nlp.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_bert_sentiment():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
    )

@st.cache_resource
def load_distilbert_sentiment():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )

@st.cache_resource
def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

CANDIDATE_LABELS = ["Pricing", "Coverage", "Enrollment",
                    "Customer Service", "Claims Processing", "Cancellation"]

def clean_text(text: str, spell, nlp) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    words = text.split()
    cache = {}
    corrected = []
    for w in words:
        if w not in cache:
            if w in spell.unknown([w]):
                sugg = spell.correction(w)
                cache[w] = sugg if sugg else w
            else:
                cache[w] = w
        corrected.append(cache[w])
    doc = nlp(" ".join(corrected))
    return " ".join([t.lemma_ for t in doc if not t.is_stop and len(t.text) > 2])

def load_df_from_upload(uploaded_files):
    dfs = []
    for f in uploaded_files:
        ext = f.name.lower()
        if ext.endswith(".xlsx") or ext.endswith(".xls"):
            dfs.append(pd.read_excel(f))
        elif ext.endswith(".csv"):
            dfs.append(pd.read_csv(f))
        elif ext.endswith(".zip"):
            with zipfile.ZipFile(f) as zf:
                for name in zf.namelist():
                    if name.endswith(".xlsx"):
                        with zf.open(name) as inner:
                            dfs.append(pd.read_excel(inner))
                    elif name.endswith(".csv"):
                        with zf.open(name) as inner:
                            dfs.append(pd.read_csv(inner))
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='padding: 8px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Insurance NLP")
    st.markdown("<hr style='border-color:#2a2a3a; margin:8px 0 20px'>", unsafe_allow_html=True)

    st.markdown("#### 📂 Charger des données")
    uploaded_files = st.file_uploader(
        "Fichiers xlsx, csv ou zip",
        type=["xlsx", "xls", "csv", "zip"],
        accept_multiple_files=True,
        help="Vos fichiers doivent contenir les colonnes 'avis_en' et 'note'."
    )

    st.markdown("<hr style='border-color:#2a2a3a; margin:20px 0'>", unsafe_allow_html=True)

    use_preloaded = st.checkbox("Utiliser données de démo", value=not bool(uploaded_files))

    st.markdown("<hr style='border-color:#2a2a3a; margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("#### ⚙️ Modèles NLP")
    run_spellcheck = st.checkbox("Correction orthographique", value=False,
                                 help="Ralentit le traitement mais améliore la qualité")
    text_col = st.text_input("Colonne texte", value="avis_en")
    note_col = st.text_input("Colonne note (1-5)", value="note")

    st.markdown("<hr style='border-color:#2a2a3a; margin:20px 0'>", unsafe_allow_html=True)
    st.caption("Insurance NLP Dashboard · v1.0")

# ─── DEMO DATA ───────────────────────────────────────────────────────────────
DEMO_REVIEWS = [
    {"avis_en": "Excellent coverage and very responsive customer service. Claims were processed quickly.", "note": 5},
    {"avis_en": "The pricing is too high compared to competitors. Not worth the cost.", "note": 2},
    {"avis_en": "Enrollment process was straightforward. Happy with the coverage options.", "note": 4},
    {"avis_en": "My claim was denied without clear explanation. Very disappointing experience.", "note": 1},
    {"avis_en": "Average service, nothing special. The policy covers what I need.", "note": 3},
    {"avis_en": "Fast claim processing and fair settlement. Would recommend to friends.", "note": 5},
    {"avis_en": "Cancellation process was a nightmare. Took months and multiple calls.", "note": 1},
    {"avis_en": "Good value for money. Customer service agents are helpful and knowledgeable.", "note": 4},
    {"avis_en": "Premium increased significantly without notice. Looking for alternatives.", "note": 2},
    {"avis_en": "Smooth enrollment and clear policy documents. Very satisfied overall.", "note": 4},
    {"avis_en": "Claims department is slow and unresponsive. Waited 3 months for settlement.", "note": 2},
    {"avis_en": "Great coverage options at competitive prices. Highly recommend.", "note": 5},
    {"avis_en": "Neutral experience. Policy is adequate but nothing outstanding.", "note": 3},
    {"avis_en": "Terrible customer support. They never answer the phone.", "note": 1},
    {"avis_en": "Very happy with the service. Claims handled professionally and fast.", "note": 5},
    {"avis_en": "Pricing is reasonable and coverage is comprehensive.", "note": 4},
    {"avis_en": "Enrollment took too long. Paperwork was excessive and confusing.", "note": 2},
    {"avis_en": "Outstanding service! The team helped me through every step of my claim.", "note": 5},
    {"avis_en": "Average experience. The coverage meets minimum requirements.", "note": 3},
    {"avis_en": "Difficult to cancel. They kept charging me after I requested cancellation.", "note": 1},
]

@st.cache_data
def make_demo_df():
    df = pd.DataFrame(DEMO_REVIEWS)
    df["text_cleaned"] = df["avis_en"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
    df["sentiment"] = df["note"].apply(note_to_sentiment)
    # Simple subject detection based on keywords
    subject_map = {
        "claim": "Claims Processing", "claims": "Claims Processing",
        "price": "Pricing", "pricing": "Pricing", "premium": "Pricing", "cost": "Pricing",
        "cover": "Coverage", "coverage": "Coverage", "policy": "Coverage",
        "enroll": "Enrollment", "enrollment": "Enrollment",
        "customer": "Customer Service", "support": "Customer Service", "service": "Customer Service",
        "cancel": "Cancellation", "cancellation": "Cancellation",
    }
    def detect_subject(text):
        text = text.lower()
        for kw, cat in subject_map.items():
            if kw in text:
                return cat
        return "Coverage"
    df["subject_category"] = df["avis_en"].apply(detect_subject)
    return df

# ─── MAIN ─────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🛡️ Insurance Review<br>NLP Dashboard</div>
    <div class="hero-subtitle">Analyse de sentiments · Prédiction de notes · Détection de sujets</div>
</div>
""", unsafe_allow_html=True)

# Load data
df = None
if uploaded_files:
    with st.spinner("Chargement des fichiers..."):
        df = load_df_from_upload(uploaded_files)
    if df is not None:
        if text_col not in df.columns:
            st.error(f"Colonne '{text_col}' introuvable. Colonnes disponibles : {list(df.columns)}")
            df = None
        else:
            df.dropna(subset=[text_col], inplace=True)
            df = df[df[text_col].str.len() > 2]
            df.reset_index(drop=True, inplace=True)
            if note_col in df.columns:
                df.dropna(subset=[note_col], inplace=True)
                df["sentiment"] = df[note_col].apply(note_to_sentiment)
            if "text_cleaned" not in df.columns:
                df["text_cleaned"] = df[text_col].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
            if "subject_category" not in df.columns:
                df["subject_category"] = "Unknown"
            st.success(f"✅ {len(df)} avis chargés depuis vos fichiers.")

if df is None or use_preloaded:
    df = make_demo_df()
    if not uploaded_files:
        st.info("💡 Mode démo — chargez vos propres fichiers dans la barre latérale.")

# ─── METRICS ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
n_pos = (df["sentiment"] == "Positif").sum() if "sentiment" in df.columns else 0
n_neu = (df["sentiment"] == "Neutre").sum() if "sentiment" in df.columns else 0
n_neg = (df["sentiment"] == "Négatif").sum() if "sentiment" in df.columns else 0
avg_note = df[note_col].mean() if note_col in df.columns else 0

metrics = [
    (len(df), "Avis total"),
    (f"{avg_note:.2f}" if avg_note else "–", "Note moyenne"),
    (n_pos, "😊 Positifs"),
    (n_neu, "😐 Neutres"),
    (n_neg, "😞 Négatifs"),
]
for col_st, (val, label) in zip([col1, col2, col3, col4, col5], metrics):
    with col_st:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble",
    "☁️ Nuage de mots",
    "🔮 Prédire une note",
    "💬 Analyser un avis",
    "📋 Données brutes",
])

# ─────────────────────── TAB 1 : VUE D'ENSEMBLE ──────────────────────────────
with tab1:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-title">Distribution des notes</div>', unsafe_allow_html=True)
        if note_col in df.columns:
            note_counts = df[note_col].value_counts().sort_index()
            fig_notes = go.Figure(go.Bar(
                x=[str(int(n)) for n in note_counts.index],
                y=note_counts.values,
                marker=dict(
                    color=note_counts.values,
                    colorscale=[[0, "#1a3a5c"], [0.5, "#3b82f6"], [1, "#63b3ed"]],
                    line=dict(color="#0f0f14", width=2),
                ),
                text=note_counts.values,
                textposition="outside",
                textfont=dict(color="#e8e4dc", family="DM Mono"),
            ))
            fig_notes.update_layout(**PLOTLY_LAYOUT, height=320,
                                    xaxis_title="Note (1 à 5)", yaxis_title="Nombre d'avis")
            st.plotly_chart(fig_notes, use_container_width=True)
        else:
            st.warning(f"Colonne '{note_col}' introuvable dans les données.")

    with col_b:
        st.markdown('<div class="section-title">Répartition des sentiments</div>', unsafe_allow_html=True)
        if "sentiment" in df.columns:
            sent_counts = df["sentiment"].value_counts()
            fig_sent = go.Figure(go.Pie(
                labels=sent_counts.index.tolist(),
                values=sent_counts.values.tolist(),
                hole=0.55,
                marker=dict(
                    colors=[PALETTE_SENT.get(s, "#888") for s in sent_counts.index],
                    line=dict(color="#0f0f14", width=3),
                ),
                textfont=dict(family="DM Mono", color="#e8e4dc"),
            ))
            fig_sent.update_layout(**PLOTLY_LAYOUT, height=320,
                                   showlegend=True,
                                   legend=dict(font=dict(family="DM Mono", color="#e8e4dc")))
            st.plotly_chart(fig_sent, use_container_width=True)

    # Top 20 mots
    st.markdown('<div class="section-title">Top 20 mots les plus fréquents</div>', unsafe_allow_html=True)
    if "text_cleaned" in df.columns:
        all_words = " ".join(df["text_cleaned"].astype(str)).split()
        word_freq = Counter(w for w in all_words if len(w) > 2)
        common = word_freq.most_common(20)
        words_list = [w for w, _ in common]
        counts_list = [c for _, c in common]

        fig_words = go.Figure(go.Bar(
            x=counts_list[::-1],
            y=words_list[::-1],
            orientation="h",
            marker=dict(
                color=counts_list[::-1],
                colorscale=[[0, "#1a3a5c"], [1, "#63b3ed"]],
                line=dict(color="#0f0f14", width=1),
            ),
            text=counts_list[::-1],
            textposition="outside",
            textfont=dict(family="DM Mono", color="#e8e4dc", size=11),
        ))
        fig_words.update_layout(**PLOTLY_LAYOUT, height=480,
                                xaxis_title="Fréquence", yaxis_title="",
                                yaxis=dict(tickfont=dict(family="DM Mono", size=12)))
        st.plotly_chart(fig_words, use_container_width=True)

    # Sujets
    if "subject_category" in df.columns and df["subject_category"].nunique() > 1:
        st.markdown('<div class="section-title">Détection de sujets</div>', unsafe_allow_html=True)
        col_s1, col_s2 = st.columns([1, 1], gap="large")
        with col_s1:
            subj_counts = df["subject_category"].value_counts()
            fig_subj = go.Figure(go.Bar(
                x=subj_counts.index.tolist(),
                y=subj_counts.values.tolist(),
                marker=dict(color=PALETTE_BLUE[:len(subj_counts)],
                            line=dict(color="#0f0f14", width=2)),
                text=subj_counts.values.tolist(),
                textposition="outside",
                textfont=dict(color="#e8e4dc", family="DM Mono"),
            ))
            fig_subj.update_layout(**PLOTLY_LAYOUT, height=320,
                                   xaxis_title="Catégorie", yaxis_title="Nombre d'avis",
                                   xaxis=dict(tickangle=-25))
            st.plotly_chart(fig_subj, use_container_width=True)

        with col_s2:
            if note_col in df.columns:
                subj_note = df.groupby("subject_category")[note_col].mean().sort_values()
                fig_subj_note = go.Figure(go.Bar(
                    x=subj_note.values,
                    y=subj_note.index.tolist(),
                    orientation="h",
                    marker=dict(
                        color=subj_note.values,
                        colorscale=[[0, "#ff6b6b"], [0.5, "#ffbd59"], [1, "#48c78e"]],
                        cmin=1, cmax=5,
                        line=dict(color="#0f0f14", width=1),
                    ),
                    text=[f"{v:.2f}" for v in subj_note.values],
                    textposition="outside",
                    textfont=dict(family="DM Mono", color="#e8e4dc"),
                ))
                fig_subj_note.update_layout(**PLOTLY_LAYOUT, height=320,
                                            xaxis_title="Note moyenne", xaxis=dict(range=[0, 5.5]))
                st.plotly_chart(fig_subj_note, use_container_width=True)

# ─────────────────────── TAB 2 : NUAGE DE MOTS ───────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Nuage de mots</div>', unsafe_allow_html=True)

    col_wc1, col_wc2 = st.columns([2, 1])
    with col_wc2:
        colormap_choice = st.selectbox("Palette", ["plasma", "viridis", "magma", "cool", "Blues"])
        max_words_wc = st.slider("Nombre de mots max", 30, 200, 100)
        filter_sentiment_wc = st.selectbox("Filtrer par sentiment",
                                           ["Tous", "Positif", "Neutre", "Négatif"])

    with col_wc1:
        try:
            from wordcloud import WordCloud
            subset = df
            if filter_sentiment_wc != "Tous" and "sentiment" in df.columns:
                subset = df[df["sentiment"] == filter_sentiment_wc]
            full_text = " ".join(subset["text_cleaned"].astype(str))
            if len(full_text.strip()) > 10:
                wc = WordCloud(
                    width=900, height=420,
                    background_color=None,
                    mode="RGBA",
                    colormap=colormap_choice,
                    max_words=max_words_wc,
                    prefer_horizontal=0.8,
                ).generate(full_text)
                fig_wc, ax_wc = plt.subplots(figsize=(9, 4.2))
                fig_wc.patch.set_facecolor("none")
                ax_wc.set_facecolor("none")
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Pas assez de texte pour générer un nuage de mots.")
        except ImportError:
            st.warning("Package `wordcloud` non installé. Installez-le avec : `pip install wordcloud`")

# ─────────────────────── TAB 3 : PRÉDIRE UNE NOTE ────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Prédiction de note (TF-IDF)</div>', unsafe_allow_html=True)
    st.caption("Le modèle TF-IDF + Régression Logistique prédit une note de 1 à 5 étoiles.")

    col_p1, col_p2 = st.columns([2, 1], gap="large")
    with col_p1:
        input_review = st.text_area(
            "Saisissez un avis client (en anglais)",
            placeholder="e.g. The claims process was incredibly slow and frustrating...",
            height=150,
        )
        predict_btn = st.button("🔮 Prédire la note", use_container_width=True, type="primary")

    with col_p2:
        st.markdown('<div class="prediction-box" id="pred-box">', unsafe_allow_html=True)
        placeholder_pred = st.empty()
        placeholder_pred.markdown("""
        <div style="text-align:center; color:#555; padding: 20px 0;">
            <div style="font-size:3rem">🔮</div>
            <div style="font-family:'DM Mono'; font-size:0.85rem; margin-top:8px;">
                En attente d'un avis...
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn and input_review.strip():
        with st.spinner("Analyse en cours..."):
            # Try to load saved model first
            tfidf_model = load_tfidf_model()

            if tfidf_model is None:
                # Train on demo data inline
                from sklearn.pipeline import Pipeline
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                demo_df = make_demo_df()
                X_train_demo = demo_df["text_cleaned"].tolist()
                y_train_demo = demo_df["note"].tolist()
                tfidf_model = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
                    ("clf", LogisticRegression(max_iter=1000)),
                ])
                tfidf_model.fit(X_train_demo, y_train_demo)

            # Clean & predict
            clean = input_review.lower()
            clean = re.sub(r"[^a-z ]", " ", clean)
            predicted_note = int(tfidf_model.predict([clean])[0])
            sentiment = note_to_sentiment(predicted_note)
            badge_class = {"Positif": "badge-pos", "Neutre": "badge-neu", "Négatif": "badge-neg"}[sentiment]

        placeholder_pred.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:3rem">{stars(predicted_note)}</div>
            <div class="stars-score">{predicted_note} / 5</div>
            <div style="margin-top:12px;">
                <span class="{badge_class}">{sentiment}</span>
            </div>
            <div style="font-family:'DM Mono'; font-size:0.75rem; color:#555; margin-top:12px;">
                Modèle : TF-IDF + LogReg
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show proba bar chart if probabilities available
        try:
            proba = tfidf_model.predict_proba([clean])[0]
            classes = tfidf_model.classes_
            fig_proba = go.Figure(go.Bar(
                x=[f"⭐ {int(c)}" for c in classes],
                y=proba,
                marker=dict(
                    color=proba,
                    colorscale=[[0, "#1a3a5c"], [1, "#63b3ed"]],
                    line=dict(color="#0f0f14", width=2),
                ),
                text=[f"{p:.0%}" for p in proba],
                textposition="outside",
                textfont=dict(color="#e8e4dc", family="DM Mono"),
            ))
            fig_proba.update_layout(**PLOTLY_LAYOUT, height=260,
                                    title="Probabilité par note",
                                    yaxis=dict(tickformat=".0%", range=[0, max(proba) * 1.3]))
            st.plotly_chart(fig_proba, use_container_width=True)
        except Exception:
            pass

# ─────────────────────── TAB 4 : ANALYSER UN AVIS ────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Analyse de sentiment en temps réel</div>', unsafe_allow_html=True)
    st.caption("Entrez un avis pour obtenir une analyse de sentiment instantanée.")

    input_sa = st.text_area(
        "Avis à analyser",
        placeholder="e.g. Great insurance company, very helpful staff and fast claims...",
        height=130,
        key="sa_input",
    )
    model_choice = st.radio("Méthode d'analyse",
                             ["Règles simples (rapide)", "DistilBERT (IA)"],
                             horizontal=True)
    analyze_btn = st.button("💬 Analyser le sentiment", use_container_width=True, type="primary")

    if analyze_btn and input_sa.strip():
        with st.spinner("Analyse en cours..."):
            if model_choice == "Règles simples (rapide)":
                pos_kw = ["great", "excellent", "fast", "helpful", "recommend", "happy",
                          "satisfied", "good", "outstanding", "love", "perfect", "amazing"]
                neg_kw = ["terrible", "awful", "slow", "nightmare", "frustrating", "denied",
                          "disappointment", "bad", "horrible", "worst", "useless", "avoid"]
                text_lower = input_sa.lower()
                pos_hits = sum(1 for k in pos_kw if k in text_lower)
                neg_hits = sum(1 for k in neg_kw if k in text_lower)
                if pos_hits > neg_hits:
                    label, score = "Positif", min(0.5 + 0.1 * pos_hits, 0.99)
                elif neg_hits > pos_hits:
                    label, score = "Négatif", min(0.5 + 0.1 * neg_hits, 0.99)
                else:
                    label, score = "Neutre", 0.6
                model_name_display = "Règles lexicales"
            else:
                try:
                    distilbert = load_distilbert_sentiment()
                    result = distilbert(input_sa[:512], truncation=True)[0]
                    raw_label = result["label"].lower()
                    score = result["score"]
                    mapping = {"positive": "Positif", "neutral": "Neutre", "negative": "Négatif"}
                    label = mapping.get(raw_label, "Neutre")
                    model_name_display = "DistilBERT multilingue"
                except Exception as e:
                    st.error(f"Erreur modèle IA : {e}. Utilisation des règles simples.")
                    label, score = "Neutre", 0.5
                    model_name_display = "Règles lexicales (fallback)"

        badge_class = {"Positif": "badge-pos", "Neutre": "badge-neu", "Négatif": "badge-neg"}[label]
        emoji = {"Positif": "😊", "Neutre": "😐", "Négatif": "😞"}[label]
        color = {"Positif": "#48c78e", "Neutre": "#ffbd59", "Négatif": "#ff6b6b"}[label]

        col_r1, col_r2 = st.columns([1, 2], gap="large")
        with col_r1:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:3.5rem">{emoji}</div>
                <div style="font-family:'Syne'; font-size:1.6rem; font-weight:800;
                            color:{color}; margin-top:8px;">{label}</div>
                <div style="font-family:'DM Mono'; font-size:0.85rem; color:#888;
                            margin-top:6px;">Confiance : {score:.0%}</div>
                <div style="font-family:'DM Mono'; font-size:0.72rem; color:#555;
                            margin-top:10px;">Modèle : {model_name_display}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            # Gauge chart for confidence
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Score de confiance", "font": {"family": "Syne", "color": "#e8e4dc", "size": 14}},
                number={"suffix": "%", "font": {"family": "Syne", "color": color, "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555",
                             "tickfont": {"family": "DM Mono", "color": "#888"}},
                    "bar": {"color": color, "thickness": 0.25},
                    "bgcolor": "#1a1a26",
                    "bordercolor": "#2a2a3a",
                    "steps": [
                        {"range": [0, 50], "color": "#1a1a26"},
                        {"range": [50, 75], "color": "#1e1e2e"},
                        {"range": [75, 100], "color": "#22223a"},
                    ],
                    "threshold": {"line": {"color": color, "width": 3}, "value": score * 100},
                },
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    height=250, margin=dict(l=20, r=20, t=40, b=10),
                                    font=dict(color="#e8e4dc"))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────── TAB 5 : DONNÉES BRUTES ──────────────────────────────
with tab5:
    st.markdown('<div class="section-title">Données brutes</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_note = st.multiselect("Filtrer par note",
                                     sorted(df[note_col].unique().tolist()) if note_col in df.columns else [],
                                     default=[])
    with col_f2:
        filter_sent = st.multiselect("Filtrer par sentiment",
                                     ["Positif", "Neutre", "Négatif"], default=[])
    with col_f3:
        filter_subj = st.multiselect("Filtrer par sujet",
                                     df["subject_category"].unique().tolist()
                                     if "subject_category" in df.columns else [],
                                     default=[])

    df_filtered = df.copy()
    if filter_note and note_col in df.columns:
        df_filtered = df_filtered[df_filtered[note_col].isin(filter_note)]
    if filter_sent and "sentiment" in df.columns:
        df_filtered = df_filtered[df_filtered["sentiment"].isin(filter_sent)]
    if filter_subj and "subject_category" in df.columns:
        df_filtered = df_filtered[df_filtered["subject_category"].isin(filter_subj)]

    st.caption(f"{len(df_filtered)} avis affichés")

    cols_to_show = [c for c in [text_col, note_col, "sentiment", "subject_category", "text_cleaned"]
                    if c in df_filtered.columns]
    st.dataframe(
        df_filtered[cols_to_show].reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    # Download
    csv_bytes = df_filtered[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger (CSV)",
        data=csv_bytes,
        file_name="avis_analyses.csv",
        mime="text/csv",
    )
