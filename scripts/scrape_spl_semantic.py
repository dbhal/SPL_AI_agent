"""
SPL Website Scraper — Semantic Chunking Version
=================================================
Improvements over basic scraper:
1. Semantic chunking — splits by meaning not word count
2. Each topic/section gets its own complete chunk
3. Better text cleaning
4. Deduplication of similar chunks

Run:
    python scripts/scrape_spl_semantic.py
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://www.spl.ise.vt.edu"
OUTPUT_FILE = "data/spl_chunks.json"
DELAY       = 1.0

SEED_URLS = [
    "/",
    "/about.html",
    "/about/people.html",
    "/about/lab-alumni.html",
    "/research.html",
    "/research/funded/past/summary.html",
    "/research/funded/active/reports.html",
    "/research/publications/journal.html",
    "/partnership.html",
    "/education.html",
    "/news-events.html",
    "/outreach.html",
    "/in-memory-of-ken-harmon.html",
    "/news-events/napw-2025.html",
    "/news-events/the-future-of-next--empowering-youth-in-decision-science-and-hum.html",
    "https://www.ise.vt.edu/people/faculty/triantis.html",
    "https://www.ise.vt.edu/people/faculty/godfrey.html",
]

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.strip()
    return text


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator=" ", strip=True) if main else ""
    return clean_text(text)


# ── Semantic chunking ─────────────────────────────────────────────────────────

def semantic_chunk(text: str, url: str, model: SentenceTransformer,
                   similarity_threshold: float = 0.5,
                   min_chunk_words: int = 30,
                   max_chunk_words: int = 400) -> list[dict]:
    """
    Split text into semantically coherent chunks.

    How it works:
    1. Split text into sentences
    2. Embed each sentence
    3. Calculate similarity between consecutive sentences
    4. When similarity drops below threshold = topic changed = new chunk
    """

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return []

    if len(sentences) < 3:
        # Too few sentences — return as single chunk
        return [{
            "text": text,
            "source_url": url,
            "chunk_index": 0,
            "word_count": len(text.split()),
        }]

    # Embed all sentences at once (faster than one by one)
    embeddings = model.encode(sentences, show_progress_bar=False)

    # Calculate cosine similarity between consecutive sentences
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_sim(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Group sentences into chunks based on similarity
    chunks = []
    current_sentences = [sentences[0]]

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        current_word_count = sum(len(s.split()) for s in current_sentences)

        if (sim < similarity_threshold or current_word_count >= max_chunk_words) \
                and current_word_count >= min_chunk_words:
            # Topic changed or chunk too big — start new chunk
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "source_url": url,
                "chunk_index": len(chunks),
                "word_count": len(chunk_text.split()),
            })
            current_sentences = [next_sentence]
        else:
            current_sentences.append(next_sentence)

    # Add final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if len(chunk_text.split()) >= min_chunk_words:
            chunks.append({
                "text": chunk_text,
                "source_url": url,
                "chunk_index": len(chunks),
                "word_count": len(chunk_text.split()),
            })

    return chunks


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: list[dict],
                       model: SentenceTransformer,
                       threshold: float = 0.95) -> list[dict]:
    """
    Remove chunks that are too similar to each other.
    Happens when multiple pages have the same nav/footer text.
    """
    if not chunks:
        return chunks

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    keep = [True] * len(chunks)

    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            if sim > threshold:
                keep[j] = False  # remove duplicate

    unique = [c for c, k in zip(chunks, keep) if k]
    removed = len(chunks) - len(unique)
    if removed > 0:
        print(f"  Removed {removed} duplicate chunks")
    return unique


# ── Link discovery ────────────────────────────────────────────────────────────

def discover_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        parsed = urlparse(full)
        if parsed.netloc == urlparse(base_url).netloc:
            if parsed.path.endswith(".html") or parsed.path.endswith("/"):
                links.add(full.split("#")[0])
    return list(links)


# ── Main scrape ───────────────────────────────────────────────────────────────

def scrape():
    import os
    os.makedirs("data", exist_ok=True)

    print("Loading embedding model for semantic chunking...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    visited = set()
    to_visit = [urljoin(BASE_URL, p) for p in SEED_URLS]
    all_chunks = []

    print(f"Starting semantic scrape of {BASE_URL}\n")

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"Fetching: {url}")
        try:
            resp = requests.get(url, timeout=10,
                                headers={"User-Agent": "SPL-Bot/1.0"})
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        text = extract_text(html)
        if len(text.split()) < 50:
            print(f"  [SKIP] Too little text")
            continue

        # Semantic chunking
        chunks = semantic_chunk(text, url, model)
        all_chunks.extend(chunks)
        print(f" -> {len(chunks)} semantic chunks")

        # Discover new links
        new_links = discover_links(html, BASE_URL)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        time.sleep(DELAY)

    # Deduplicate
    print(f"\nDeduplicating {len(all_chunks)} chunks...")
    all_chunks = deduplicate_chunks(all_chunks, model)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n Done! {len(all_chunks)} semantic chunks from {len(visited)} pages")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape()