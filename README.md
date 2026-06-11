# 📰 NewsLens — AI-Powered News Summarization & Analysis

> End-to-end NLP pipeline: paste any news URL → get an abstractive summary, semantic keywords, and per-sentence sentiment analysis in an interactive dashboard.

🔗 **Live Demo:** [newslens-ai-powered-news-summarization-sentiment-analysis-azsx.streamlit.app](https://newslens-ai-powered-news-summarization-sentiment-analysis-azsx.streamlit.app)

![Pipeline Architecture](pipeline_final.png)

---

## ✨ What It Does

Paste a news article URL or text → click **Run Full Pipeline** → get:

| Output | Detail |
|---|---|
| 📝 **Abstractive Summary** | BART generates new sentences, not just extracts |
| 🔑 **Top Keywords** | KeyBERT semantic scoring + bar chart |
| 💬 **Sentiment Score** | RoBERTa — Positive / Neutral / Negative with confidence % |
| 🔍 **Per-Sentence Breakdown** | Every sentence scored individually |
| 🗞️ **Article Metadata** | Title, author, publish date, top image |
| ⏱️ **Pipeline Timings** | Time taken at each stage |

---

## 📊 Model Performance

| Model | Size | Avg Compression | GPU Speed | CPU Speed | Cloud |
|---|---|---|---|---|---|
| `facebook/bart-large-cnn` | 1.6 GB | ~70% | ~2s | ~18s | ⚠️ Out of memory |
| `sshleifer/distilbart-cnn-12-6` | 900 MB | ~68% | ~1s | ~10s | ✅ Recommended |
| `google/pegasus-cnn_dailymail` | 2.2 GB | ~72% | ~3s | ~25s | ⚠️ Out of memory |

> ⚠️ Streamlit Cloud has 1 GB RAM — use **DistilBART** for cloud deployment. BART-large and PEGASUS work locally with GPU.

---

## 🏗️ Pipeline Architecture

```
News URL / Pasted Text
        │
        ▼
┌───────────────┐
│    Scraper    │  newspaper4k → title, author, date, image, text
│  BS4 fallback │
└───────┬───────┘
        │ Clean text
        ▼
┌───────────────┐
│ Text Cleaner  │  Strip ads, normalize whitespace, fix quotes
└───────┬───────┘
        │
        ▼
┌───────────────┐
│     BART      │  model.generate() — GPU fp16 — auto-chunks long docs
│  Summarizer   │
└───────┬───────┘
        │
   ┌────┴────┐
   ▼         ▼
┌────────┐ ┌──────────┐
│KeyBERT │ │ RoBERTa  │
│Keywords│ │Sentiment │
└────────┘ └──────────┘
      │         │
      └────┬────┘
           ▼
  ┌─────────────────┐
  │   Streamlit     │  Summary · Keywords · Sentiment · Timings
  │   Dashboard     │
  └─────────────────┘
```

---

## 🚀 How to Use the App

### Using the Live Demo

1. Open the **[live app](https://newslens-ai-powered-news-summarization-sentiment-analysis-azsx.streamlit.app)**
2. **Select DistilBART** from the model dropdown (recommended for Cloud)
3. Either:
   - Paste a news URL in the input box, **or**
   - Toggle **✍️ Paste article text** and paste the article directly
4. Click **🚀 Run Full Pipeline**
5. Scroll down to see the full dashboard

> 💡 **Tip:** If a URL fails to scrape (some sites block bots), use the paste toggle instead.

---

### Sidebar Controls

| Control | What it does | Recommended |
|---|---|---|
| **Model** | Switch summarization model | DistilBART on Cloud, BART locally |
| **Max tokens** | Max length of summary | 130 |
| **Min tokens** | Prevents too-short summaries | 30 |
| **Beam width** | Higher = better quality, slower | 4 |
| **Length penalty** | >1 = longer, <1 = shorter | 1.0 |
| **No-repeat n-gram** | Prevents repetition | 3 |
| **Keywords** | How many keyphrases to extract | 10 |
| **KeyBERT toggle** | Semantic vs TF-IDF extraction | On locally, Off on Cloud |

---

## 🛠️ Run Locally

### Requirements
- Python 3.10+
- NVIDIA GPU recommended (CPU works, slower)

### Step 1 — Clone

```bash
git clone https://github.com/Ishita-1408/NewsLens-AI-Powered-News-Summarization-Sentiment-Analysis.git
cd NewsLens-AI-Powered-News-Summarization-Sentiment-Analysis
```

### Step 2 — Virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install PyTorch

Check your CUDA version first: `nvidia-smi`

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only / Mac
pip install torch torchvision torchaudio
```

### Step 4 — Install dependencies

```bash
pip install --only-binary=:all: transformers accelerate sentencepiece protobuf
pip install newspaper4k beautifulsoup4 requests lxml keybert sentence-transformers textblob plotly streamlit
python -m textblob.download_corpora
```

### Step 5 — Run

```bash
streamlit run app.py
```

Open **http://localhost:8501**

> ⏳ First run downloads ~2.5 GB of models. Cached permanently after that.

---

## 🐳 Docker

```bash
docker build -t newslens .
docker run -p 8501:8501 newslens

# With GPU
docker run --gpus all -p 8501:8501 newslens
```

Open **http://localhost:8501**

---

## 📁 Project Structure

```
NewsLens/
├── app.py              # Streamlit dashboard + pipeline orchestrator
├── scraper.py          # Stage 2: URL scraping & text cleaning
├── summarizer.py       # Stage 3: BART direct inference (no pipeline())
├── keywords.py         # Stage 4: KeyBERT / TF-IDF keyword extraction
├── sentiment.py        # Stage 5: RoBERTa sentiment analysis
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies for Streamlit Cloud
├── Dockerfile
├── .dockerignore
├── pipeline_final.png  # Architecture diagram
├── .streamlit/
│   └── config.toml     # Dark theme
└── README.md
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Web framework | Streamlit |
| Summarization | facebook/bart-large-cnn, DistilBART, PEGASUS |
| Deep learning | PyTorch 2.x |
| NLP library | Hugging Face Transformers 5.x |
| Scraping | newspaper4k + BeautifulSoup4 |
| Keywords | KeyBERT + sentence-transformers |
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment |
| Charts | Plotly |
| Containerisation | Docker |

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: plotly` | `pip install plotly` |
| `Neither newspaper4k nor beautifulsoup4 installed` | `pip install newspaper4k beautifulsoup4` |
| `Unknown task summarization` | Replace `app.py` with latest version from repo |
| Scraper fails on a URL | Toggle **✍️ Paste article text** and paste manually |
| App stuck on "Extracting keywords…" | Disable KeyBERT toggle in sidebar (uses TF-IDF instead) |
| Out of memory on Cloud | Switch to DistilBART model |
| Out of GPU memory locally | Set beam width to 1–2 or switch to DistilBART |

---

## 👩‍💻 Author

**Ishita** 

---

## 📜 License

MIT — free to use, modify, and deploy.
