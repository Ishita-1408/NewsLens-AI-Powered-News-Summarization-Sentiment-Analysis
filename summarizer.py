"""
Stage 3: BART Summarizer
Uses direct model inference — no pipeline() call.
Works with all transformers versions including 5.x
"""
import gc
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = {
    "facebook/bart-large-cnn":       {"label": "BART-large-CNN (best quality)", "max_input": 1024},
    "sshleifer/distilbart-cnn-12-6": {"label": "DistilBART (faster)",           "max_input": 1024},
    "google/pegasus-cnn_dailymail":  {"label": "PEGASUS (alternative)",         "max_input": 1024},
}


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    dtype     = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model


def _chunk_text(text: str, max_words: int = 700) -> list:
    """Split long articles into overlapping chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, i, overlap = [], 0, 50
    while i < len(words):
        chunks.append(" ".join(words[i: i + max_words]))
        i += max_words - overlap
    return chunks


def _summarize_chunk(
    text: str,
    tokenizer,
    model,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> str:
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize(
    text: str,
    model_name: str,
    max_length: int = 130,
    min_length: int = 30,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 3,
) -> dict:
    tokenizer, model = load_summarizer(model_name)
    chunks = _chunk_text(text)

    summaries = []
    for chunk in chunks:
        s = _summarize_chunk(
            chunk, tokenizer, model,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        summaries.append(s)

    combined = " ".join(summaries)

    # If multiple chunks, do one final merge pass
    if len(chunks) > 1:
        combined = _summarize_chunk(
            combined, tokenizer, model,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    # Free memory after inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "summary": combined,
        "chunks":  len(chunks),
        "words":   len(combined.split()),
    }