# ==============================
# utils/subject.py
# ==============================
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_similarity(text, model, label_embs, labels):
    emb = model.encode([text])
    sims = cosine_similarity(emb, label_embs)[0]
    pairs = list(zip(labels, sims))
    return sorted(pairs, key=lambda x: x[1], reverse=True)