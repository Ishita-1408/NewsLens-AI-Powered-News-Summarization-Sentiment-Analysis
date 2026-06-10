"""
Stage 4: Keyword Extraction
TF-IDF only — no downloads, no hanging, instant results.
KeyBERT can be re-enabled once all-MiniLM-L6-v2 is fully cached.
"""
import re
from collections import Counter
import streamlit as st

STOPWORDS = set("a about above after again against all am an and any are as at be because been before being below between both but by can cannot could did do does doing don't down during each few for from further get got has have having he her here him himself his how i if in into is it its itself let me more most my myself no nor not of off on once only or other our out over own same she should so some such than that the their theirs them then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself said also just like many told according year years one two three new".split())


def _tfidf(text, top_n=10):
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
             if w not in STOPWORDS]
    if not words:
        return []
    tf    = Counter(words)
    total = len(words)
    scores = {w: c / total for w, c in tf.items()}
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    for bg, c in Counter(bigrams).most_common(30):
        scores[bg] = (c / len(bigrams)) * 1.5
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(k, v) for k, v in ranked
            if 2 <= len(k.split()) <= 3 or len(k) >= 4][:top_n]


def extract_keywords(text, top_n=10, use_keybert=True):
    # TF-IDF only — instant, no model download needed
    raw = _tfidf(text, top_n)
    mx  = raw[0][1] if raw else 1
    return [{"keyword": kw, "score": round(s / mx, 3),
             "method": "TF-IDF"} for kw, s in raw]