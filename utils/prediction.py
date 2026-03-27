# ==============================
# utils/prediction.py
# ==============================
def predict_all(text, model, transformer, task="rating"):
    tf_pred = model.predict([text])[0]
    tf_conf = max(model.predict_proba([text])[0])

    bert_res = transformer(text)[0]
    bert_pred = bert_res['label']
    bert_conf = bert_res['score']

    return tf_pred, tf_conf, bert_pred, bert_conf