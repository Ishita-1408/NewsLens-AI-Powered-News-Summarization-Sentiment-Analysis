import re, requests
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse
try:
    from newspaper import Article
    NEWSPAPER_OK = True
except ImportError:
    NEWSPAPER_OK = False
try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

@dataclass
class ScrapedArticle:
    url: str
    title: str = ""
    authors: list = field(default_factory=list)
    publish_date: Optional[str] = None
    raw_text: str = ""
    clean_text: str = ""
    top_image: str = ""
    source_domain: str = ""
    word_count: int = 0
    sentence_count: int = 0
    error: Optional[str] = None

def _clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 25 and s.isupper(): continue
        if len(s) < 8: continue
        lines.append(s)
    text = ' '.join(lines)
    text = text.replace('\u2018',"'").replace('\u2019',"'").replace('\u201c','"').replace('\u201d','"')
    return text.strip()

def _fallback_bs4(url):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsSummarizer/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.find("title")
    title_text = title.get_text(strip=True) if title else ""
    container = (soup.find("article") or soup.find("main")
                 or soup.find("div", {"class": re.compile(r"article|content|story|body", re.I)})
                 or soup.body)
    paragraphs = container.find_all("p") if container else soup.find_all("p")
    raw = "\n".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text()) > 40)
    return title_text, raw

def scrape(url):
    result = ScrapedArticle(url=url)
    result.source_domain = urlparse(url).netloc.replace("www.", "")
    try:
        if NEWSPAPER_OK:
            art = Article(url)
            art.download(); art.parse()
            result.title = art.title or ""
            result.authors = art.authors or []
            result.publish_date = str(art.publish_date) if art.publish_date else None
            result.top_image = art.top_image or ""
            result.raw_text = art.text or ""
        elif BS4_OK:
            result.title, result.raw_text = _fallback_bs4(url)
        else:
            raise RuntimeError("Neither newspaper4k nor beautifulsoup4 installed.")
        if len(result.raw_text.strip()) < 100:
            raise ValueError("Scraped text too short — site may block scrapers.")
        result.clean_text = _clean_text(result.raw_text)
        result.word_count = len(result.clean_text.split())
        result.sentence_count = len(re.split(r'(?<=[.!?])\s+', result.clean_text))
    except Exception as e:
        result.error = str(e)
    return result
