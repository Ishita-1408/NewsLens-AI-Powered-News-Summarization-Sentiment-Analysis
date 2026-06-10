"""
Stage 4: Keyword Extraction
KeyBERT with hard timeout + TF-IDF fallback.
If KeyBERT hangs or fails for any reason, TF-IDF runs instantly.
"""
import re
import threading
from collections import Counter
import streamlit as st

try:
    from keybert import KeyBERT
    KEYBERT_OK = True
except ImportError:
    KEYBERT_OK = False

STOPWORDS = set("a about above after again against all am an and any are as at be because been before being below between both but by can cannot could did do does doing don't down during each few for from further get got has have having he her here him himself his how i if in into is it its itself let me more most my myself no nor not of off on once only or other our out over own same she should so some such than that the their theirs them then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself said also just like many told according year years one two three new".split())

# ── KeyBERT loader with timeout ───────────────────────────────────────────────
_kb_instance = None
_kb_loaded    = False
_kb_failed    = False

def _load_kb_thread():
    global _kb_instance, _kb_loaded, _kb_failed
    try:
        _kb_instance = KeyBERT(model="all-MiniLM-L6-v2")
        _kb_loaded   = True
    except Exception:
        _kb_failed = True

@st.cache_resource(show_spinner=False)
def _get_keybert():
    """Returns KeyBERT instance if loaded within 30s, else None → TF-IDF fallback."""
    global _kb_instance, _kb_loaded, _kb_failed
    if not KEYBERT_OK:
        return None
    if _kb_loaded:
        return _kb_instance
    if _kb_failed:
        return None
    # Try loading in a thread with a 30-second timeout
    t = threading.Thread(target=_load_kb_thread, daemon=True)
    t.start()
    t.join(timeout=30)           # wait max 30 seconds
    if _kb_loaded:
        return _kb_instance
    return None                  # timed out → use TF-IDF


# ── TF-IDF fallback — no downloads, always instant ───────────────────────────
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


# ── Public API ────────────────────────────────────────────────────────────────
def extract_keywords(text, top_n=10, use_keybert=True):
    kb = _get_keybert() if use_keybert else None

    if kb is not None:
        try:
            raw = kb.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                use_maxsum=True,
                nr_candidates=30,
                top_n=top_n,
            )
            if raw:
                return [{"keyword": kw, "score": round(s, 3),
                         "method": "KeyBERT (semantic)"} for kw, s in raw]
        except Exception:
            pass  # fall through to TF-IDF

    # TF-IDF — guaranteed to return results
    raw = _tfidf(text, top_n)
    mx  = raw[0][1] if raw else 1
    return [{"keyword": kw, "score": round(s / mx, 3),
             "method": "TF-IDF (fallback)"} for kw, s in raw]