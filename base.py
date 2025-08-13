import os
import faiss
import pickle
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_community.utilities import GoogleSerperAPIWrapper

# Constants
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
VECTOR_DB_PATH = "vector_store.faiss"
INDEX_METADATA_PATH = "doc_metadata.pkl"
CONTENT_TYPES = ["blogs", "articles", "case studies", "strategies", "ebooks"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_API_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_API_KEY"),
    host="https://cloud.langfuse.com",
) 

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def serper_search(topic, max_results_per_type=5):
    trace = langfuse.trace(
        name="serper_search",
        input={"topic": topic, "max_results_per_type": max_results_per_type}
    )
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    search = GoogleSerperAPIWrapper()
    all_results = []

    for ctype in CONTENT_TYPES:
        query = f"{topic} {ctype}"
        print(f"[LangChain Serper] Searching: {query}")

        search_span = trace.span(
            name="search_content_type",
            input={"query": query, "type": ctype}
        )
        try:
            search_result = search.results(query)
            organic = search_result.get("organic", [])
            urls = []
            for r in organic[:max_results_per_type]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                link = r.get("link", "")
                link_span = search_span.span(
                    name="search_result",
                    input={"title": title, "link": link, "snippet": snippet}
                )
                link_span.end()
                urls.append(link)
                all_results.append((title, snippet, link, ctype))
            
        except Exception as e:
            search_span.output = {"error": str(e)}
            print(f"[ERROR] Failed search for '{query}': {e}")
        search_span.end()

    

    text_blocks = []
    metadata_blocks = []

    for title, snippet, url, ctype in all_results:
        print(f"[Processing] {ctype.upper()} → {url}")

        # scrape_span = trace.span(name="scrape_url", input=url)
        text = extract_text_from_url(url)
        # scrape_span.update(output={"text_length": len(text) if text else 0})
        # scrape_span.end()

        if not text or len(text) < 300:
            text = f"{title}\n{snippet}"

        if len(text) >= 300:
            from textwrap import wrap
            chunks = wrap(text, 800)
            for chunk in chunks:
                text_blocks.append(chunk)
                metadata_blocks.append({
                    "type": ctype,
                    "source": url,
                    "summary": chunk[:300]
                })

    if text_blocks:
        print("[Vector Store] Storing extracted documents...")
        # store_span = trace.span(name="store_vector_db", input={"num_blocks": len(text_blocks)})
        store_in_vector_db(text_blocks, metadata_blocks)
        # store_span.update(output={"status": "stored"})
        # store_span.end()
    else:
        print("[Warning] No valid content found.")
        # trace.update(output={"warning": "no_valid_content"})

    

def extract_text_from_url(url, timeout=10):
    try:
        import requests
        res = requests.get(url, headers=HEADERS, timeout=timeout)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        return text
    except Exception as e:
        print(f"[ERROR] Scraping failed at {url}: {e}")
        return None

def store_in_vector_db(text_blocks, metadata_blocks):
    if os.path.exists(VECTOR_DB_PATH):
        index = faiss.read_index(VECTOR_DB_PATH)
        with open(INDEX_METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)
        metadata = []

    embeddings = model.encode(text_blocks)
    index.add(np.array(embeddings).astype("float32"))
    metadata.extend(metadata_blocks)

    faiss.write_index(index, VECTOR_DB_PATH)
    with open(INDEX_METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

from collections import defaultdict
from textwrap import shorten

def query_vector_db(user_query, top_k=10, chunk_limit=500):
    """
    Query FAISS DB and return concise, merged results to reduce LLM token usage.

    Args:
        user_query (str): Search query.
        top_k (int): Number of unique documents to return.
        chunk_limit (int): Max characters per document summary.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        return ["[ERROR] Vector DB is empty. Please run a search first."]

    # Load FAISS index and metadata
    index = faiss.read_index(VECTOR_DB_PATH)
    with open(INDEX_METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Encode query & search (over-fetch for deduplication)
    query_embedding = model.encode([user_query]).astype("float32")
    D, I = index.search(query_embedding, top_k * 3)

    # Merge chunks per source
    docs = defaultdict(list)
    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            docs[entry['source']].append(entry['summary'])

    # Combine & trim
    results = []
    for source, chunks in docs.items():
        # Keep order stable
        combined_text = " ".join(chunks)
        # Shorten for token efficiency
        concise_text = shorten(combined_text, width=chunk_limit, placeholder="...")
        results.append(f"{source}\n\n{concise_text}")

    # Return only top_k unique documents
    return results[:top_k]





