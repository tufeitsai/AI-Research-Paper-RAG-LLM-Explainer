import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pickle

TEXT_DIR = "../data/extracted_texts"
EMBEDDING_OUTPUT = "../data/embeddings/chunks.pkl"

os.makedirs("data/embeddings", exist_ok=True)

# embedding models
model = SentenceTransformer("all-MiniLM-L6-v2")

# chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)


all_chunks = []

for filename in os.listdir(TEXT_DIR):
    if not filename.endswith(".txt"):
        continue
    paper_id = filename.replace(".txt", "")
    with open(os.path.join(TEXT_DIR, filename), "r", encoding="utf-8") as f:
        text = f.read()

    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        chunk_data = {
            "paper_id": paper_id,
            "chunk_id": f"{paper_id}_{i}",
            "text": chunk
        }
        all_chunks.append(chunk_data)

print(f"🔢 Total chunks: {len(all_chunks)}")

# Embed all chunks
texts = [chunk["text"] for chunk in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Attach embeddings
for i in range(len(all_chunks)):
    all_chunks[i]["embedding"] = embeddings[i]

with open(EMBEDDING_OUTPUT, "wb") as f:
    pickle.dump(all_chunks, f)

print(f"\n✅ Saved chunk embeddings to {EMBEDDING_OUTPUT}")
