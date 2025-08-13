# AI-Research-Paper-RAG-LLM-Explainer

## Overview
This project is a Retrieval-Augmented Generation (RAG) pipeline for answering research questions based on AI research papers.  
It uses a local **LLaMA3** model (via Ollama) and integrates **FAISS** for vector search over processed paper chunks.

---

## Features
- Ingest and chunk research paper PDFs
- Generate embeddings and store in FAISS vector index
- Query the RAG system with or without baseline comparison
- Support for **MMR** (Maximal Marginal Relevance) retrieval
- Local **LLaMA3** inference via **Ollama**

---

## Data Download
The processed data files (PDFs, embeddings, FAISS index) are too large for GitHub.  
They are hosted on Google Drive and can be downloaded here:

**[Download Data Folder from Google Drive](<https://drive.google.com/file/d/19Wj3eormBvAdIdgBlWihnfo1gJQAOAaY/view?usp=sharing>)**

After downloading, place the `data` folder in the project root before running any scripts.

---

## Installation

### 1 Clone the repository
```bash
git clone https://github.com/<your_username>/AI-Research-Paper-RAG-LLM-Explainer.git
cd AI-Research-Paper-RAG-LLM-Explainer
```

### 2 Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3 Usage
Run Baseline Query (No RAG)
```bash
python scripts/query_rag.py --q "Your question here" --baseline
```
Run RAG Query
```bash
python scripts/query_rag.py --q "Your question here" --mmr --fetch_k 200
```

### Example
Question:
What evaluation benchmarks are most commonly used to assess large language models in recent research papers?

Baseline Output (No RAG):
1. Perplexity (PPL)
2. Accuracy (ACC) and F1-score
3. BLEU score
4. ROUGE score
5. METEOR score
6. Consistency (CONS)

RAG Output:
1. COMET - benchmark for evaluating code-trained LLMs
2. ComplexCodeEval - complex code evaluation
3. Idiom translation benchmarks
4. Multilingual performance benchmarks

