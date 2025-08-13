# scripts/query_rag.py
import os, json, argparse, re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Config: Ollama + Paths
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # local model
USE_PY_CLIENT = True

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH  = os.path.join(INDEX_DIR, "meta.npy")
TXT_PATH   = os.path.join(INDEX_DIR, "texts.npy")
PAPERS_JSON = os.path.join(DATA_DIR, "arxiv_papers.json")


# Load index + metadata
def load_index_and_artifacts():
    index = faiss.read_index(INDEX_PATH)
    meta  = np.load(META_PATH, allow_pickle=True)  # chunk_id per vector
    texts = np.load(TXT_PATH, allow_pickle=True)   # raw text per vector
    id2paper, chunk_paper_ids = {}, []

    # Map paper_id -> {title, url}
    if os.path.exists(PAPERS_JSON):
        with open(PAPERS_JSON, "r", encoding="utf-8") as f:
            for p in json.load(f):
                arxiv_id = p["id"]
                title    = p["title"].strip()
                url      = p.get("pdf_url") or p.get("link") or f"https://arxiv.org/abs/{arxiv_id}"
                id2paper[arxiv_id] = {"title": title, "url": url}


    for chunk_id in meta:
        pid = chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id
        chunk_paper_ids.append(pid)

    return index, meta, texts, id2paper, np.array(chunk_paper_ids, dtype=object)

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def _paper_from_chunk_id(chunk_id: str):
    return chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id


# Optional BM25 (hybrid)
_BM25 = None
def init_bm25(texts):
    global _BM25
    from rank_bm25 import BM25Okapi
    tokenized = [t.lower().split() for t in texts]
    _BM25 = BM25Okapi(tokenized)

def bm25_scores(query: str, top_n: int = 200):
    from rank_bm25 import BM25Okapi
    if _BM25 is None:
        return []
    toks = query.lower().split()
    scores = _BM25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:top_n]
    return [(int(i), float(scores[int(i)])) for i in idxs]


# Retrieval variants
def retrieve_topk(query: str, k: int, index, meta, texts):
    q = EMBEDDER.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return list(zip(I[0], D[0]))

def retrieve_mmr(query: str, k: int, fetch_k: int, index, meta, texts):
    q = EMBEDDER.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, fetch_k)
    cand_idx, cand_scores = I[0], D[0]

    # re-embed candidates
    cand_embs = EMBEDDER.encode([texts[i] for i in cand_idx]).astype("float32")
    faiss.normalize_L2(cand_embs)
    qv = q[0]

    selected, selected_set = [], set()
    while len(selected) < min(k, len(cand_idx)):
        best_pos, best_score = -1, -1e9
        for pos, i in enumerate(cand_idx):
            if i in selected_set:
                continue
            sim_q = float(np.dot(cand_embs[pos], qv))
            sim_sel = 0.0
            if selected:
                sel_vecs = cand_embs[[list(cand_idx).index(s) for s in selected]]
                sim_sel = float(np.max(sel_vecs @ cand_embs[pos]))
            mmr = 0.5 * sim_q - 0.5 * sim_sel
            if mmr > best_score:
                best_score, best_pos = mmr, pos
        chosen = int(cand_idx[best_pos])
        selected.append(chosen); selected_set.add(chosen)
    # return indices + original faiss scores for logging
    idx_to_score = {int(i): float(s) for i, s in zip(cand_idx, cand_scores)}
    return [(i, idx_to_score[i]) for i in selected]

def title_boost_rerank(pairs, chunk_paper_ids, id2paper, title_keywords):
    """Boost items whose paper title matches any keyword."""
    if not title_keywords:
        return pairs
    kws = [kw.strip().lower() for kw in title_keywords if kw.strip()]
    boosted = []
    for idx, score in pairs:
        pid  = chunk_paper_ids[idx]
        info = id2paper.get(pid, {})
        title = info.get("title", "")
        bonus = 0.0
        if title:
            t = title.lower()
            if any(kw in t for kw in kws):
                bonus = 0.15
        boosted.append((idx, score + bonus))
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted

def package_results(pairs, meta, texts, id2paper, limit):
    out = []
    for idx, score in pairs[:limit]:
        chunk_id = meta[idx]
        pid = _paper_from_chunk_id(chunk_id)
        paper = id2paper.get(pid, {})
        out.append({
            "chunk_id": chunk_id,
            "paper_id": pid,
            "score": float(score),
            "text": texts[idx],
            "title": paper.get("title", f"(unknown title: {pid})"),
            "url": paper.get("url", f"https://arxiv.org/abs/{pid}")
        })
    return out

def retrieve_hybrid(query: str, k: int, fetch_k: int, index, meta, texts, id2paper, chunk_paper_ids,
                    use_mmr: bool, use_hybrid: bool, title_keywords=None, paper_id=None):
    # Optional single-paper restriction
    if paper_id:
        base_pairs = retrieve_mmr(query, k=fetch_k, fetch_k=max(fetch_k, 200), index=index, meta=meta, texts=texts) \
                     if use_mmr else retrieve_topk(query, k=fetch_k, index=index, meta=meta, texts=texts)
        filtered = [(i, s) for (i, s) in base_pairs if chunk_paper_ids[i] == paper_id]
        if not filtered:
            return []
        filtered.sort(key=lambda x: x[1], reverse=True)
        return package_results(filtered, meta, texts, id2paper, k)

    base_pairs = retrieve_mmr(query, k=k, fetch_k=fetch_k, index=index, meta=meta, texts=texts) \
                 if use_mmr else retrieve_topk(query, k=fetch_k, index=index, meta=meta, texts=texts)

    # Optionally mix in BM25 candidates
    if use_hybrid:
        bm = bm25_scores(query, top_n=fetch_k)
        # merge by index with simple score fusion (normalize roughly)
        bm_norm = {i: (r / (max([b for _, b in bm]) + 1e-9)) * 0.3 for i, r in bm}  # weight 0.3
        dense = dict(base_pairs)
        for i, s in bm_norm.items():
            dense[i] = dense.get(i, 0.0) + s
        base_pairs = sorted(dense.items(), key=lambda x: x[1], reverse=True)

    # title boost
    base_pairs = title_boost_rerank(base_pairs, chunk_paper_ids, id2paper, title_keywords)

    # Package top-k
    return package_results(base_pairs, meta, texts, id2paper, k)


# Prompts
def build_rag_prompt(question: str, contexts, max_chars=12000):
    blocks, total = [], 0
    for i, c in enumerate(contexts, 1):
        chunk = f"[{i}] {c['title']} ({c['url']})\n{c['text']}"
        if total + len(chunk) > max_chars:
            break
        blocks.append(chunk); total += len(chunk)
    context_block = "\n\n---\n\n".join(blocks)
    system = ("You are an AI research-paper explainer. Use ONLY the provided context. "
              "Answer clearly for a data science audience. Cite sources inline like [1], [2]. "
              "If the answer is not present, say you’re unsure.")
    user = f"Question: {question}\n\nContext:\n{context_block}\n\nAnswer:"
    return system, user

def build_baseline_prompt(question: str):
    system = ("You are an AI research-paper expert. Answer concisely and accurately. "
              "If unsure, say so.")
    user = f"Question: {question}\n\nAnswer:"
    return system, user


# Ollama client
def answer_with_ollama(system_prompt, user_prompt, model: str):
    if USE_PY_CLIENT:
        import ollama
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            options={"temperature": 0.2}
        )
        return resp["message"]["content"].strip()
    else:
        import requests
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


# CLI
def main():
    parser = argparse.ArgumentParser(description="Query RAG (or baseline) using local Ollama")
    parser.add_argument("--q", required=True, help="Your question")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks in final prompt")
    parser.add_argument("--fetch_k", type=int, default=120, help="Candidates to fetch before re-ranking")
    parser.add_argument("--model", default=None, help="Ollama model (e.g., llama3:8b-instruct)")
    parser.add_argument("--baseline", action="store_true", help="Baseline: no retrieval/context")
    parser.add_argument("--mmr", action="store_true", help="Diversify retrieval (MMR)")
    parser.add_argument("--hybrid", action="store_true", help="Use BM25 + FAISS (requires rank-bm25)")
    parser.add_argument("--title_keywords", default="", help="Comma-separated keywords to boost in paper titles (e.g., 'llama,llama 2')")
    parser.add_argument("--paper_id", default=None, help="Restrict to a single paper_id (e.g., 2307.XXXXvY)")
    args = parser.parse_args()

    model = args.model or OLLAMA_MODEL

    if args.baseline:
        system, user = build_baseline_prompt(args.q)
        print(f"🔎 Baseline query: {args.q}\n🧠 Model: {model}")
        ans = answer_with_ollama(system, user, model)
        print("\n💬 Answer (baseline):\n"); print(ans)
        return

    index, meta, texts, id2paper, chunk_paper_ids = load_index_and_artifacts()
    if args.hybrid:
        try:
            init_bm25(texts)
        except Exception as e:
            print(f"⚠️ BM25 init failed ({e}); continuing without hybrid.")
            args.hybrid = False

    title_kw = [s for s in args.title_keywords.split(",")] if args.title_keywords else []
    hits = retrieve_hybrid(
        query=args.q,
        k=args.k,
        fetch_k=args.fetch_k,
        index=index, meta=meta, texts=texts,
        id2paper=id2paper, chunk_paper_ids=chunk_paper_ids,
        use_mmr=args.mmr, use_hybrid=args.hybrid,
        title_keywords=title_kw, paper_id=args.paper_id
    )

    print(f"🔎 RAG query: {args.q}  | top-k={args.k}  fetch_k={args.fetch_k}  mmr={args.mmr}  hybrid={args.hybrid}")
    print(f"📚 Hits ({len(hits)}):")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']:.4f} id={h['chunk_id']}")
        print(f"Title: {h['title']}\nURL:   {h['url']}")
        preview = (h['text'][:500] + "…") if len(h['text']) > 500 else h['text']
        print(preview)

    if not hits:
        print("\n⚠️ No hits after filtering. Try increasing --fetch_k, removing --paper_id, or loosening keywords.")
        return

    system, user = build_rag_prompt(args.q, hits)
    print("\n🧠 Generating with Ollama (RAG)…")
    ans = answer_with_ollama(system, user, model)
    print("\n💬 Answer (RAG):\n"); print(ans)

if __name__ == "__main__":
    main()
