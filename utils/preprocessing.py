# ==============================
# utils/preprocessing.py
# ==============================
import re
import spacy
from spellchecker import SpellChecker

spell = SpellChecker(language='en')
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

cache = {}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


def clean_text_nlp(text):
    if not isinstance(text, str) or text.strip() == "": return ""

    text = text.lower()
    text = re.sub(r'[^a-z ]', ' ', text)

    # Correction orthographique ultra-rapide via cache
    words = text.split()
    corrected = []
    for w in words:
        if w not in cache:
            if w in spell.unknown([w]):
                sugg = spell.correction(w)
                cache[w] = sugg if sugg is not None else w
            else:
                cache[w] = w
        corrected.append(cache[w])

    # Lemmatisation SpaCy
    doc = nlp(" ".join(corrected))
    return " ".join([t.lemma_ for t in doc if not t.is_stop and len(t.text) > 2])