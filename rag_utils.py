import os, json, glob, math
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
from tqdm import tqdm
from pypdf import PdfReader
import docx
import markdown

EMBED_MODEL = "text-embedding-3-small"   # cost-effective
GENERATE_MODEL = "gpt-4o-mini"           # for answering
CHUNK_TOKENS = 600                       # ~rough chunking target
CHUNK_OVERLAP = 120

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _read_md(path: str) -> str:
    return _read_txt(path)

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def _read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md", ".markdown"]:
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    # fallback: try plain text
    return _read_txt(path)

def simple_token_count(s: str) -> int:
    # crude token proxy: ~4 chars per token heuristic
    return max(1, math.ceil(len(s) / 4))

def chunk_text(text: str, target_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> List[str]:
    # very simple splitter by sentences/lines
    import re
    units = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", text)
    chunks, cur, cur_tokens = [], [], 0
    for u in units:
        t = simple_token_count(u)
        if cur_tokens + t > target_tokens and cur:
            chunk = " ".join(cur).strip()
            if chunk:
                chunks.append(chunk)
            # overlap: keep tail
            if overlap > 0 and chunk:
                tail = chunk.split()[-overlap*4:]  # rough char→token proxy
                cur, cur_tokens = [" ".join(tail)], simple_token_count(" ".join(tail))
            else:
                cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += t
    final = " ".join(cur).strip()
    if final:
        chunks.append(final)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    # batch to reduce API calls; keep it simple here
    vectors = []
    for t in tqdm(texts, desc="Embedding", unit="chunk"):
        emb = client.embeddings.create(model=EMBED_MODEL, input=t).data[0].embedding
        vectors.append(emb)
    return np.array(vectors, dtype="float32")

def save_index(index, meta: List[Dict], path_idx="index/faiss.index", path_meta="index/meta.json"):
    os.makedirs(os.path.dirname(path_idx), exist_ok=True)
    faiss.write_index(index, path_idx)
    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_index(path_idx="index/faiss.index", path_meta="index/meta.json") -> Tuple[faiss.Index, List[Dict]]:
    index = faiss.read_index(path_idx)
    with open(path_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

def build_or_rebuild_index(data_glob="data/**/*.*") -> None:
    files = [p for p in glob.glob(data_glob, recursive=True) if os.path.isfile(p)]
    docs_meta, all_chunks = [], []

    print(f"Found {len(files)} files.")
    for path in files:
        try:
            raw = load_document(path)
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue
        chunks = chunk_text(raw)
        for i, c in enumerate(chunks):
            docs_meta.append({"source": path, "chunk_id": i, "text": c})
            all_chunks.append(c)

    if not all_chunks:
        raise RuntimeError("No text chunks found. Add files to ./data")

    vecs = embed_texts(all_chunks)  # (N, d)
    d = vecs.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vecs)
    save_index(index, docs_meta)
    print(f"Indexed {len(all_chunks)} chunks from {len(files)} files.")

def search(query: str, k) -> List[Dict]:
    index, meta = load_index()
    qvec = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    q = np.array([qvec], dtype="float32")
    D, I = index.search(q, k)
    results = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        m = meta[int(idx)]
        results.append({
            "rank": int(rank+1),
            "distance": float(dist),
            "source": m["source"],
            "chunk_id": m["chunk_id"],
            "text": m["text"]
        })
    return results

def answer(query: str, contexts: List[Dict]) -> str:
    # Build a slim prompt
    context_blocks = "\n\n---\n\n".join(
        f"[{i+1}] Source: {c['source']} (chunk {c['chunk_id']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    system = (
        "You are a concise assistant for personal information retrieval. "
        "Answer strictly using the provided context. If unsure, say you don't know."
    )
    user = f"Question: {query}\n\nContext:\n{context_blocks}\n\nAnswer succinctly:"
    resp = client.chat.completions.create(
        model=GENERATE_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
