"""
Deep News Summarizer — Full 7-Stage Pipeline
URL → Scrape → Clean → BART → Keywords → Sentiment → Dashboard
No pipeline() call for summarization — uses model.generate() directly.
"""
import time, re
import streamlit as st
import torch
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from scraper   import scrape, ScrapedArticle
from keywords  import extract_keywords
from sentiment import analyze_sentiment, EMOJI_MAP, COLOR_MAP

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Deep News Summarizer", page_icon="📰",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html,body,.stApp{background:#0d0f18;color:#c8cde8}
section[data-testid="stSidebar"]{background:#13162a;border-right:1px solid #252a45}
.card{background:linear-gradient(135deg,#161929,#1c2038);border:1px solid #252a45;border-radius:14px;padding:20px 24px;margin-bottom:16px}
.card-title{font-size:.75rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#5a61a0;margin-bottom:10px}
.hero{font-size:2.2rem;font-weight:900;background:linear-gradient(90deg,#7c83f7,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{color:#6b72a8;font-size:.95rem;margin-bottom:20px}
.kw-chip{display:inline-block;background:#1e2545;border:1px solid #3d4a8a;border-radius:20px;padding:3px 12px;font-size:.8rem;color:#a5b4fc;margin:3px;font-weight:500}
.sent-row{border-left:3px solid #333;padding:6px 12px;margin:4px 0;background:#161929;border-radius:0 8px 8px 0;font-size:.85rem}
.stat-big{font-size:1.9rem;font-weight:800;color:#7c83f7}
.stat-label{font-size:.72rem;color:#6b72a8;text-transform:uppercase;letter-spacing:.06em}
.meta-item{color:#8b92b3;font-size:.82rem;margin:4px 0}
hr{border-color:#1e2240!important}
.stButton>button{border-radius:10px!important;font-weight:600!important}
</style>
""", unsafe_allow_html=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "facebook/bart-large-cnn":       "BART-large-CNN (best quality)",
    "sshleifer/distilbart-cnn-12-6": "DistilBART (faster)",
    "google/pegasus-cnn_dailymail":  "PEGASUS (alternative)",
}

# ── Load model — NO pipeline() call ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model

# ── Direct inference — NO pipeline() ─────────────────────────────────────────
def run_inference(text, tokenizer, model, max_new_tokens, min_new_tokens,
                  num_beams, length_penalty, no_repeat_ngram_size):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", max_length=1024,
                       truncation=True, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def chunk_text(text, max_words=700):
    words = text.split()
    if len(words) <= max_words: return [text]
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        i += max_words - 50
    return chunks

def do_summarize(text, model_name, max_len, min_len, num_beams, lp, ng):
    tokenizer, model = load_model(model_name)
    chunks = chunk_text(text)
    parts = [run_inference(c, tokenizer, model, max_len, min_len, num_beams, lp, ng)
             for c in chunks]
    combined = " ".join(parts)
    if len(chunks) > 1:
        combined = run_inference(combined, tokenizer, model, max_len, min_len, num_beams, lp, ng)
    return {"summary": combined, "chunks": len(chunks), "words": len(combined.split())}

# ── Helpers ───────────────────────────────────────────────────────────────────
def compression(orig, summ):
    ow = len(orig.split())
    return round((1 - len(summ.split()) / ow) * 100, 1) if ow else 0.0

def reading_time(text):
    return f"{max(1, round(len(text.split())/200))} min read"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")
    st.divider()
    st.markdown("### 🤖 Model")
    model_name = st.selectbox("Summarization model", list(MODELS.keys()),
                          index=1,   # ← defaults to DistilBART
                          format_func=lambda x: MODELS[x])
    st.markdown("### 📏 Output Length")
    max_len = st.slider("Max tokens", 60, 300, 130, 10)
    min_len = st.slider("Min tokens", 10, 100, 30, 5)
    if min_len >= max_len: min_len = max_len - 20
    st.markdown("### 🎛 Decoding")
    num_beams       = st.slider("Beam width", 1, 8, 4)
    length_penalty  = st.slider("Length penalty", 0.5, 2.0, 1.0, 0.1)
    no_repeat_ngram = st.slider("No-repeat n-gram", 0, 4, 3)
    st.markdown("### 🔑 Keywords")
    top_n_kw    = st.slider("Keywords to extract", 5, 20, 10)
    use_keybert = st.toggle("Use KeyBERT (semantic)", value=True)
    st.divider()
    st.markdown("### 🖥 Hardware")
    if torch.cuda.is_available():
        st.success(f"🟢 GPU — {torch.cuda.get_device_name(0)}")
    else:
        st.warning("🟡 CPU")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">📰 Deep News Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">End-to-end pipeline: Scrape → Clean → BART → Keywords → Sentiment → Dashboard</div>', unsafe_allow_html=True)
STAGES = ["🔗 URL Input","🕷 Scraper","🧹 Cleaner","🤖 BART","🔑 Keywords","💬 Sentiment","📊 Dashboard"]
st.markdown("<div style='margin-bottom:18px;line-height:2.4'>"+" <span style='color:#2d3258'>→</span> ".join(STAGES)+"</div>", unsafe_allow_html=True)
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
manual_mode = st.toggle("✍️ Paste article text (instead of URL)", value=False)
if manual_mode:
    manual_text = st.text_area("Paste your news article here", height=220,
                               placeholder="Paste any news article text…")
    url_input = ""
else:
    url_input   = st.text_input("🔗 News Article URL",
                                placeholder="https://www.bbc.com/news/…")
    manual_text = ""

run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)
st.divider()

# ── Pipeline ──────────────────────────────────────────────────────────────────
if run_btn:
    if not url_input and not manual_text:
        st.warning("⚠️ Please enter a URL or paste article text.")
        st.stop()

    R, T = {}, {}
    prog = st.progress(0, "Starting…")

    # Stage 1+2 — Scrape
    prog.progress(10, "🕷 Scraping & cleaning…")
    t0 = time.time()
    if manual_text:
        art = ScrapedArticle(url=url_input or "manual-input")
        art.clean_text     = manual_text
        art.raw_text       = manual_text
        art.word_count     = len(manual_text.split())
        art.sentence_count = len(re.split(r'(?<=[.!?])\s+', manual_text.strip()))
        art.title          = "Manual Input"
    else:
        art = scrape(url_input)
    T["scrape"] = round(time.time()-t0, 2)

    if art.error:
        st.error(f"❌ Scraper error: {art.error}")
        st.info("Try the '✍️ Paste article text' toggle.")
        st.stop()
    R["article"] = art
    prog.progress(28, "✅ Scraped")

    # Stage 3 — BART (direct model.generate)
    prog.progress(35, "🤖 Running BART…")
    t0 = time.time()
    with st.spinner(f"Summarizing with {MODELS[model_name]}…"):
        summ = do_summarize(art.clean_text, model_name,
                            max_len, min_len, num_beams,
                            length_penalty, no_repeat_ngram)
    T["summarize"] = round(time.time()-t0, 2)
    R["summary"] = summ
    prog.progress(58, "✅ Summary done")

    # Stage 4 — Keywords
    prog.progress(62, "🔑 Extracting keywords…")
    t0 = time.time()
    kws = extract_keywords(art.clean_text, top_n=top_n_kw, use_keybert=use_keybert)
    T["keywords"] = round(time.time()-t0, 2)
    R["keywords"] = kws
    prog.progress(78, "✅ Keywords done")

    # Stage 5 — Sentiment
    prog.progress(82, "💬 Analyzing sentiment…")
    t0 = time.time()
    sent = analyze_sentiment(art.clean_text)
    T["sentiment"] = round(time.time()-t0, 2)
    R["sentiment"] = sent
    prog.progress(100, "✅ Pipeline complete!")
    time.sleep(0.3); prog.empty()

    st.session_state.update({"R": R, "T": T, "MN": model_name})

# ── Dashboard ─────────────────────────────────────────────────────────────────
if "R" in st.session_state:
    R, T, MN = st.session_state["R"], st.session_state["T"], st.session_state["MN"]
    art  = R["article"]
    summ = R["summary"]
    kws  = R["keywords"]
    sent = R["sentiment"]
    cr   = compression(art.clean_text, summ["summary"])

    st.markdown("## 📊 Analysis Dashboard")

    # KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    for col,val,lbl in [
        (k1, art.word_count,                               "Source Words"),
        (k2, summ["words"],                                "Summary Words"),
        (k3, f"{cr}%",                                     "Compression"),
        (k4, f"{sent['emoji']} {sent['overall']['label']}", "Sentiment"),
        (k5, f"{sum(T.values()):.1f}s",                    "Total Time"),
    ]:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;padding:16px">'
                        f'<div class="stat-big">{val}</div>'
                        f'<div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)

    # Metadata
    with st.expander("🗞️ Article Metadata", expanded=True):
        c1,c2 = st.columns([2,1])
        with c1:
            if art.title and art.title != "Manual Input":
                st.markdown(f"### {art.title}")
            if art.authors:
                st.markdown(f"<div class='meta-item'>✍️ {', '.join(art.authors)}</div>", unsafe_allow_html=True)
            if art.publish_date:
                st.markdown(f"<div class='meta-item'>📅 {art.publish_date}</div>", unsafe_allow_html=True)
            if art.source_domain:
                st.markdown(f"<div class='meta-item'>🌐 {art.source_domain}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta-item'>⏱ {reading_time(art.clean_text)} → ~1 min summary</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta-item'>📝 {art.word_count} words · {art.sentence_count} sentences</div>", unsafe_allow_html=True)
        with c2:
            if art.top_image:
                st.image(art.top_image, use_column_width=True)

    st.divider()

    # Summary
    st.markdown("### 🤖 BART Summary")
    st.markdown(f'<div class="card"><div class="card-title">Generated Summary — {MODELS[MN]}</div>'
                f'<p style="font-size:1.05rem;line-height:1.75;color:#d4d8f0">{summ["summary"]}</p></div>',
                unsafe_allow_html=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    for col,lbl,val in [(mc1,"Model",MN.split("/")[-1]),(mc2,"Chunks",summ["chunks"]),
                         (mc3,"Beams",num_beams),(mc4,"Time",f"{T['summarize']}s")]:
        with col: st.caption(lbl); st.markdown(f"**{val}**")
    with st.expander("📋 Copy summary"):
        st.code(summ["summary"], language=None)

    st.divider()

    # Keywords + Sentiment
    kw_col, sent_col = st.columns([1,1], gap="large")

    with kw_col:
        st.markdown("### 🔑 Top Keywords")
        st.caption(f"Extracted via: {kws[0]['method'] if kws else '—'}")
        st.markdown("".join(f'<span class="kw-chip">#{k["keyword"]}</span>' for k in kws), unsafe_allow_html=True)
        if kws:
            fig = go.Figure(go.Bar(
                x=[k["score"] for k in kws], y=[k["keyword"] for k in kws],
                orientation="h",
                marker=dict(color=[k["score"] for k in kws],
                            colorscale=[[0,"#2d3a8a"],[1,"#7c83f7"]], line=dict(width=0)),
                text=[f'{k["score"]:.3f}' for k in kws], textposition="outside",
            ))
            fig.update_layout(height=360, margin=dict(l=0,r=50,t=10,b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis=dict(autorange="reversed", tickfont=dict(color="#a5b4fc",size=12)),
                              xaxis=dict(tickfont=dict(color="#6b72a8"), showgrid=False),
                              font=dict(color="#c8cde8"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with sent_col:
        st.markdown("### 💬 Sentiment Analysis")
        ov = sent["overall"]
        st.caption(f"Model: {ov['method']}")
        gc = sent["color"]
        st.markdown(f'<div class="card" style="text-align:center;padding:22px;border-color:{gc}44">'
                    f'<div style="font-size:3rem">{sent["emoji"]}</div>'
                    f'<div style="font-size:1.7rem;font-weight:800;color:{gc}">{ov["label"]}</div>'
                    f'<div style="color:#8b92b3;margin-top:6px">Confidence: <strong style="color:{gc}">'
                    f'{round(ov["confidence"]*100,1)}%</strong></div>'
                    f'<div style="color:#5a61a0;font-size:.78rem;margin-top:4px">{ov["method"]}</div>'
                    f'</div>', unsafe_allow_html=True)
        dist = sent["distribution"]
        if sum(dist.values()) > 0:
            fig2 = go.Figure(go.Pie(
                labels=list(dist.keys()), values=list(dist.values()), hole=0.55,
                marker=dict(colors=["#22c55e","#eab308","#ef4444"]),
                textinfo="label+percent", textfont=dict(color="#c8cde8",size=11),
            ))
            fig2.update_layout(height=230, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                               font=dict(color="#c8cde8"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
            st.caption(f"Distribution across {sum(dist.values())} sentences")

    st.divider()

    # Per-sentence sentiment
    if sent["sentences"]:
        with st.expander(f"🔍 Per-Sentence Sentiment ({len(sent['sentences'])} sentences)"):
            for s in sent["sentences"]:
                color = COLOR_MAP[s["label"]]
                emoji = EMOJI_MAP[s["label"]]
                st.markdown(
                    f'<div class="sent-row" style="border-left-color:{color}">'
                    f'<span style="color:{color};font-weight:600">{emoji} {s["label"]}</span>'
                    f'<span style="color:#5a61a0;font-size:.75rem;margin-left:8px">{round(s["confidence"]*100,1)}%</span><br>'
                    f'<span style="color:#c8cde8">{s["sentence"]}</span></div>',
                    unsafe_allow_html=True)

    # Original text
    with st.expander("📄 View cleaned article text"):
        st.text_area("", art.clean_text, height=280, label_visibility="collapsed")

    # Timings
    st.divider()
    st.markdown("### ⏱ Pipeline Timings")
    icons = {"scrape":"🕷","summarize":"🤖","keywords":"🔑","sentiment":"💬"}
    for col,(stage,sec) in zip(st.columns(len(T)), T.items()):
        with col:
            st.markdown(f'<div class="card" style="text-align:center;padding:12px 8px">'
                        f'<div style="font-size:1.4rem">{icons.get(stage,"⚙️")}</div>'
                        f'<div style="font-size:1.1rem;font-weight:700;color:#7c83f7">{sec}s</div>'
                        f'<div style="font-size:.72rem;color:#6b72a8;text-transform:uppercase">{stage}</div>'
                        f'</div>', unsafe_allow_html=True)

    # History
    if "history" not in st.session_state: st.session_state.history = []
    st.session_state.history.insert(0, {
        "title":       art.title or art.url[:60],
        "summary":     summ["summary"][:200]+"…",
        "sentiment":   f"{sent['emoji']} {sent['overall']['label']}",
        "compression": cr,
    })
    st.session_state.history = st.session_state.history[:6]

if "history" in st.session_state and len(st.session_state.history) > 1:
    st.divider()
    with st.expander(f"🕘 History ({len(st.session_state.history)} articles this session)"):
        for h in st.session_state.history:
            c1,c2,c3 = st.columns([3,1,1])
            with c1: st.markdown(f"**{h['title'][:70]}**")
            with c2: st.markdown(h["sentiment"])
            with c3: st.markdown(f"`{h['compression']}% compressed`")
            st.caption(h["summary"]); st.markdown("---")