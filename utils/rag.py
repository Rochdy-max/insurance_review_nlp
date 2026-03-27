# ==============================
# utils/rag.py
# ==============================
import requests
import json

def run_rag(question, df):
    url = "http://localhost:11434/api/generate"

    prompt = f"Extract filters from: {question}. Return JSON only."

    res = requests.post(url, json={"model": "gemma3", "prompt": prompt})
    raw = res.json()['response']

    try:
        filters = json.loads(raw)
    except:
        filters = {}

    filtered = df.copy()

    if 'sentiment' in filters:
        filtered = filtered[filtered['sentiment'] == filters['sentiment']]

    context = "\n".join(filtered['text_cleaned'].head(10).tolist())

    final_prompt = f"Answer: {question}\nContext: {context}"

    res2 = requests.post(url, json={"model": "gemma3", "prompt": final_prompt})

    return res2.json()['response']