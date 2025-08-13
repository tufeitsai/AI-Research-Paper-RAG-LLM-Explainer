# scripts/build_vector_db.py
import os
import pickle
import numpy as np
import faiss

# ----- Path helpers (run-safe from any cwd) -----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
INDEX_DIR = os.path.join(DATA_DIR, "index")

CHUNKS_PKL = os.path.join(EMB_DIR, "chunks.pkl")
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH  = os.path.join(INDEX_DIR, "meta.npy")   # chunk_id mapping
TXT_PATH   = os.path.join(INDEX_DIR, "texts.npy")  # raw text for each vector

def main():
    print(f"📦 Loading chunks from: {CHUNKS_PKL}")
    with open(CHUNKS_PKL, "rb") as f:
        chunks = pickle.load(f)

    # embeddings -> float32 array (N, D)
    X = np.array([c["embedding"] for c in chunks], dtype="float32")
    texts = np.array([c["text"] for c in chunks], dtype=object)
    meta  = np.array([c["chunk_id"] for c in chunks], dtype=object)
    faiss.normalize_L2(X)

    # build index
    d = X.shape[1]
    print(f"🧠 Building FAISS index with dim={d}, vectors={X.shape[0]}")
    index = faiss.IndexFlatIP(d)  # cosine similarity
    index.add(X)

    faiss.write_index(index, INDEX_PATH)
    np.save(META_PATH, meta, allow_pickle=True)
    np.save(TXT_PATH, texts, allow_pickle=True)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved meta  to {META_PATH}")
    print(f"Saved texts to {TXT_PATH}")

if __name__ == "__main__":
    main()
