import streamlit as st
import pandas as pd
import joblib
import spacy
from spellchecker import SpellChecker
 
# Configuration de la page
st.set_page_config(page_title="Analyse d'Avis Assurance", layout="centered")
 
# --- CHARGEMENT DES RESSOURCES ---
@st.cache_resource
def load_nlp():
    # Chargement du modèle Spacy pour le prétraitement
    try:
        return spacy.load("en_core_web_sm")
    except:
        return None
 
@st.cache_resource
def load_model():
    # Remplacez 'model.pkl' et 'vectorizer.pkl' par vos vrais fichiers exportés du notebook
    # Si vous n'avez pas encore exporté, utilisez : joblib.dump(model, 'model.pkl')
    try:
        model = joblib.load('models/tf_idf_nlp.pkl')
        # vectorizer = joblib.load('vectorizer.pkl')
        return model#, vectorizer
    except:
        return None#, None
 
nlp = load_nlp()
# mo:#del, vectorizer = load_model()
model = load_model()
spell = SpellChecker()
 
# --- FONCTION DE NETTOYAGE (Basée sur votre notebook) ---
def clean_text(text):
    if not text:
        return ""
    # Correction orthographique simple
    words = text.split()
    corrected_words = [spell.correction(w) if spell.correction(w) else w for w in words]
    text = " ".join(corrected_words)
    # Prétraitement Spacy (Lemmatisation, suppression stop words)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)
 
# --- INTERFACE UTILISATEUR ---
st.title("🛡️ Analyseur d'Avis Assurances")
st.markdown("""
Cette application prédit la note (nombre d'étoiles) ou le sentiment d'un avis client 
en utilisant les techniques de NLP développées dans le projet.
""")
 
# Zone de saisie
user_input = st.text_area("Saisissez l'avis client ici :", placeholder="Ex: Great service, very fast response...")
 
if st.button("Analyser l'avis"):
    if user_input:
        if model:# and vectorizer:
            with st.spinner('Analyse en cours...'):
                # 1. Prétraitement
                cleaned_text = clean_text(user_input)
                # 2. Vectorisation
                text_vector = [cleaned_text]
                # text_vector = vectorizer.transform([cleaned_text])
                # 3. Prédiction
                prediction = model.predict(text_vector)[0]
                # Affichage des résultats
                st.subheader("Résultat de la prédiction")
                st.write(f"**Note estimée :** {prediction} ⭐")
                # Optionnel : barre de confiance ou jauge (si le modèle supporte predict_proba)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(text_vector)
                    st.bar_chart(pd.DataFrame(probs, columns=model.classes_).T)
        else:
            st.error("Erreur : Les fichiers du modèle ('model.pkl', 'vectorizer.pkl') sont introuvables. Veuillez les exporter depuis votre notebook.")
    else:
        st.warning("Veuillez entrer un texte pour l'analyse.")
 
# --- SECTION EXPLORATION (Basée sur la partie exploration du notebook) ---
st.divider()
st.subheader("📊 Aperçu des données d'entraînement")
st.info("Cette section affiche les conclusions de l'exploration initiale effectuée dans le notebook.")
# Vous pouvez ajouter ici des graphiques statiques ou des métriques issues de votre notebook
st.write("- **Nombre total d'avis analysés :** 34,435")
st.write("- **Distribution des notes :** La majorité des avis sont soit très positifs, soit très négatifs.")