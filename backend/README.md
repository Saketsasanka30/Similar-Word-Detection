# SEMANTIQ — Backend Setup Guide

## Architecture Overview

SEMANTIQ uses a separated Frontend + REST API architecture:

1. **Integrated Live Runtime (`server.ts` + `server/nlp/`)** — Runs directly in this workspace on port `3000` with genuine NLP models:
   - **WordNet 3.1 (`wordnet-db`)**: Full Princeton WordNet 3.1 binary-offset reader computing Wu-Palmer similarity, Path similarity, Least Common Subsumer (LCS), and semantic network exploration.
   - **TF-IDF + Cosine (`server/nlp/tfidf.ts`)**: Tokenization, English stopword filtering, smooth IDF weighting over a reference linguistic corpus, and exact L2 cosine similarity.
   - **Word Embeddings (`minishlab/potion-base-8M`)**: 256-dimensional static subword/token embeddings running via ONNX (`@huggingface/transformers`).
   - **Semantic Transformer (`Xenova/all-MiniLM-L6-v2`)**: 384-dimensional contextual `sentence-transformers/all-MiniLM-L6-v2` embeddings running via ONNX (`@huggingface/transformers`).

2. **Standalone Python FastAPI Service (`backend/main.py`)** — Provided for standalone Python/Uvicorn deployments using NLTK WordNet, Scikit-learn `TfidfVectorizer`, and Python `sentence-transformers`:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
