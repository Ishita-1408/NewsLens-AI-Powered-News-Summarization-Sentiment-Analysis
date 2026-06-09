import re, torch, streamlit as st
from transformers import pipeline as hf_pipeline

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {"positive":"Positive","POSITIVE":"Positive","LABEL_2":"Positive",
             "neutral":"Neutral","NEUTRAL":"Neutral","LABEL_1":"Neutral",
             "negative":"Negative","NEGATIVE":"Negative","LABEL_0":"Negative"}
EMOJI_MAP = {"Positive":"🟢","Neutral":"🟡","Negative":"🔴"}
COLOR_MAP  = {"Positive":"#22c55e","Neutral":"#eab308","Negative":"#ef4444"}

@st.cache_resource(show_spinner=False)
def _load_sentiment_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return hf_pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device, truncation=True, max_length=512)

def _textblob_fallback(text):
    try:
        from textblob import TextBlob
        pol = TextBlob(text).sentiment.polarity
        if pol>0.05: label,conf="Positive",min(0.5+pol,1.0)
        elif pol<-0.05: label,conf="Negative",min(0.5+abs(pol),1.0)
        else: label,conf="Neutral",0.6
        return {"label":label,"confidence":round(conf,3),"polarity":round(pol,3),"method":"TextBlob (fallback)"}
    except:
        return {"label":"Neutral","confidence":0.5,"polarity":0.0,"method":"Default (no model)"}

def _sentence_sentiments(text, pipe):
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if len(s.split())>=5][:20]
    results = []
    for s in sents:
        try:
            out = pipe(s, truncation=True, max_length=128)[0]
            label = LABEL_MAP.get(out["label"], out["label"])
            results.append({"sentence": s[:140]+("…" if len(s)>140 else ""), "label":label, "confidence":round(out["score"],3)})
        except: pass
    return results

def analyze_sentiment(text):
    try:
        pipe = _load_sentiment_pipeline()
        out = pipe(text[:2000], truncation=True, max_length=512)[0]
        label = LABEL_MAP.get(out["label"], out["label"])
        overall = {"label":label,"confidence":round(out["score"],3),
                   "polarity":round(out["score"]*(1 if label=="Positive" else -1 if label=="Negative" else 0),3),
                   "method":"RoBERTa (twitter-sentiment)"}
        sentences = _sentence_sentiments(text, pipe)
    except:
        overall = _textblob_fallback(text); sentences = []
    dist = {"Positive":0,"Neutral":0,"Negative":0}
    for s in sentences: dist[s["label"]] = dist.get(s["label"],0)+1
    return {"overall":overall,"sentences":sentences,"distribution":dist,
            "emoji":EMOJI_MAP.get(overall["label"],"⚪"),"color":COLOR_MAP.get(overall["label"],"#888")}
