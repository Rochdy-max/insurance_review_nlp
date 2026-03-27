import streamlit as st
import joblib

# 1. Chargement du modèle

model = joblib.load('tf_idf_nlp.pkl')
 
st.title("Analyseur d'Avis Assurance")
 
# 2. Entrée utilisateur

texte = st.text_area("Collez l'avis client ici :")
 
if st.button("Analyser"):

    if texte:

        # 3. Prédiction de la note

        note = model.predict([texte])[0]

        st.success(f"Note prédite : {note} / 5")
 
        # 4. Explication simple (Points bonus)

        st.subheader("Explication")

        mots_cles = {

            "positifs": ["great", "easy", "fast", "good", "satisfied", "helpful"],

            "négatifs": ["bad", "slow", "expensive", "worst", "price", "never"]

        }

        mots_trouves_pos = [m for m in mots_cles["positifs"] if m in texte.lower()]

        mots_trouves_neg = [m for m in mots_cles["négatifs"] if m in texte.lower()]
 
        if mots_trouves_pos:

            st.write(f"✅ Mots positifs détectés : {', '.join(mots_trouves_pos)}")

        if mots_trouves_neg:

            st.write(f"❌ Mots négatifs détectés : {', '.join(mots_trouves_neg)}")

    else:

        st.error("Veuillez entrer du texte.")