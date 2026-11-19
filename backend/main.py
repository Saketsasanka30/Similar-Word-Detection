"""
SEMANTIQ — Standalone Python FastAPI NLP Backend Reference Implementation.
Matches the REST API contract served by server.ts across all 5 NLP algorithms:
WordNet (Wu-Palmer), TF-IDF + Cosine, Word2Vec, FastText, and Sentence-BERT.
"""

from datetime import datetime, timezone
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    wn.synsets("entity")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

app = FastAPI(title="SEMANTIQ NLP API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_transformer_model: Optional[SentenceTransformer] = None
_word2vec_model = None
_fasttext_model = None


def get_transformer() -> SentenceTransformer:
    global _transformer_model
    if _transformer_model is None:
        _transformer_model = SentenceTransformer(TRANSFORMER_MODEL_NAME)
    return _transformer_model


def get_word2vec():
    global _word2vec_model
    if _word2vec_model is None:
        import gensim.downloader as api
        _word2vec_model = api.load("word2vec-google-news-300")
    return _word2vec_model


def get_fasttext():
    global _fasttext_model
    if _fasttext_model is None:
        import gensim.downloader as api
        _fasttext_model = api.load("fasttext-wiki-news-subwords-300")
    return _fasttext_model


class WordAnalyzeRequest(BaseModel):
    word_a: str = Field(..., min_length=1, max_length=80)
    word_b: str = Field(..., min_length=1, max_length=80)
    algorithm: Literal["wordnet", "tfidf", "word2vec", "fasttext", "transformer", "all"] = "all"


class SentenceAnalyzeRequest(BaseModel):
    sentence_a: str = Field(..., min_length=3, max_length=600)
    sentence_b: str = Field(..., min_length=3, max_length=600)


class ContextAnalyzeRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=80)
    sentence_a: str = Field(..., min_length=3, max_length=600)
    sentence_b: Optional[str] = None


class BatchPairItem(BaseModel):
    id: Optional[str] = None
    word_a: str
    word_b: str


class BatchAnalyzeRequest(BaseModel):
    pairs: List[BatchPairItem]
    algorithm: Literal["wordnet", "tfidf", "word2vec", "fasttext", "transformer", "all"] = "all"


@app.get("/api/algorithms")
def algorithms_catalog():
    return {
        "algorithms": [
            {"id": "wordnet", "name": "WordNet (Wu-Palmer)", "category": "Lexical"},
            {"id": "tfidf", "name": "TF-IDF + Cosine", "category": "Statistical"},
            {"id": "word2vec", "name": "Word2Vec", "category": "Word embeddings"},
            {"id": "fasttext", "name": "FastText", "category": "Subword embeddings"},
            {"id": "transformer", "name": "Sentence-BERT", "category": "Contextual embeddings"},
        ]
    }


@app.post("/api/analyze/word")
@app.post("/api/analyze/compare")
def analyze_word(req: WordAnalyzeRequest):
    word_a = req.word_a.strip()
    word_b = req.word_b.strip()
    if not word_a or not word_b:
        raise HTTPException(status_code=400, detail="Both word_a and word_b are required.")

    results = {}
    run_all = req.algorithm == "all"

    if run_all or req.algorithm == "wordnet":
        t0 = time.perf_counter()
        syns_a = wn.synsets(word_a.replace(" ", "_"))
        syns_b = wn.synsets(word_b.replace(" ", "_"))
        best_wup = None
        for sa in syns_a:
            for sb in syns_b:
                sim = sa.wup_similarity(sb)
                if sim is not None and (best_wup is None or sim > best_wup):
                    best_wup = sim
        results["wordnet"] = {
            "score": round(best_wup * 100, 1) if best_wup is not None else None,
            "status": "completed" if best_wup is not None else "not_available",
            "processingTimeMs": round((time.perf_counter() - t0) * 1000, 1),
        }

    if run_all or req.algorithm == "tfidf":
        t0 = time.perf_counter()
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform([word_a, word_b])
        cos = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        results["tfidf"] = {
            "score": round(max(0.0, min(1.0, cos)) * 100, 1),
            "status": "completed",
            "processingTimeMs": round((time.perf_counter() - t0) * 1000, 1),
        }

    if run_all or req.algorithm == "word2vec":
        t0 = time.perf_counter()
        try:
            kv = get_word2vec()
            if word_a.lower() in kv and word_b.lower() in kv:
                cos = float(kv.similarity(word_a.lower(), word_b.lower()))
                results["word2vec"] = {
                    "score": round(max(0.0, min(1.0, cos)) * 100, 1),
                    "status": "completed",
                    "processingTimeMs": round((time.perf_counter() - t0) * 1000, 1),
                }
            else:
                results["word2vec"] = {"score": None, "status": "not_available"}
        except Exception as exc:
            results["word2vec"] = {"score": None, "status": "not_available", "reasonIfUnavailable": str(exc)}

    if run_all or req.algorithm == "fasttext":
        t0 = time.perf_counter()
        try:
            ft = get_fasttext()
            cos = float(ft.similarity(word_a.lower(), word_b.lower()))
            results["fasttext"] = {
                "score": round(max(0.0, min(1.0, cos)) * 100, 1),
                "status": "completed",
                "processingTimeMs": round((time.perf_counter() - t0) * 1000, 1),
            }
        except Exception as exc:
            results["fasttext"] = {"score": None, "status": "not_available", "reasonIfUnavailable": str(exc)}

    if run_all or req.algorithm == "transformer":
        t0 = time.perf_counter()
        model = get_transformer()
        embs = model.encode([word_a, word_b], normalize_embeddings=True)
        cos = float(util.cos_sim(embs[0], embs[1]).item())
        results["transformer"] = {
            "score": round(max(0.0, min(1.0, cos)) * 100, 1),
            "status": "completed",
            "processingTimeMs": round((time.perf_counter() - t0) * 1000, 1),
        }

    return {
        "word_a": word_a,
        "word_b": word_b,
        "algorithm": req.algorithm,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/analyze/sentence")
def analyze_sentence(req: SentenceAnalyzeRequest):
    sa = req.sentence_a.strip()
    sb = req.sentence_b.strip()
    model = get_transformer()
    embs = model.encode([sa, sb], normalize_embeddings=True)
    trans_cos = float(util.cos_sim(embs[0], embs[1]).item())

    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([sa, sb])
    tfidf_cos = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])

    return {
        "sentence_a": sa,
        "sentence_b": sb,
        "results": {
            "transformer": {"score": round(max(0.0, min(1.0, trans_cos)) * 100, 1), "status": "completed"},
            "tfidf": {"score": round(max(0.0, min(1.0, tfidf_cos)) * 100, 1), "status": "completed"},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/analyze/context")
def analyze_context(req: ContextAnalyzeRequest):
    syns = wn.synsets(req.word.strip().replace(" ", "_"))
    if not syns:
        return {"word": req.word, "found": False, "totalSenses": 0}
    model = get_transformer()
    senses = [f"{req.word}: {s.definition()}" for s in syns[:10]]
    texts = [req.sentence_a] + ([req.sentence_b] if req.sentence_b else []) + senses
    embs = model.encode(texts, normalize_embeddings=True)
    offset = 2 if req.sentence_b else 1
    scores_1 = [float(util.cos_sim(embs[0], embs[offset + i]).item()) for i in range(len(senses))]
    best_1 = int(np.argmax(scores_1))
    result = {
        "word": req.word,
        "found": True,
        "totalSenses": len(syns),
        "sentence1": {"detectedSense": {"synsetId": syns[best_1].name(), "gloss": syns[best_1].definition()}},
    }
    if req.sentence_b:
        scores_2 = [float(util.cos_sim(embs[1], embs[offset + i]).item()) for i in range(len(senses))]
        best_2 = int(np.argmax(scores_2))
        result["sentence2"] = {"detectedSense": {"synsetId": syns[best_2].name(), "gloss": syns[best_2].definition()}}
    return result


@app.post("/api/analyze/batch")
def analyze_batch(req: BatchAnalyzeRequest):
    rows = []
    for idx, p in enumerate(req.pairs):
        res = analyze_word(WordAnalyzeRequest(word_a=p.word_a, word_b=p.word_b, algorithm=req.algorithm))
        rows.append({"id": p.id or f"pair_{idx+1}", **res})
    return {"pairCount": len(rows), "results": rows}


@app.get("/api/semantic/related/{word}")
def semantic_related(word: str):
    syns = wn.synsets(word.strip().replace(" ", "_"))
    synonyms = sorted({l.name().replace("_", " ") for s in syns[:3] for l in s.lemmas()})
    hypernyms = sorted({l.name().replace("_", " ") for s in syns[:3] for h in s.hypernyms() for l in h.lemmas()})
    hyponyms = sorted({l.name().replace("_", " ") for s in syns[:3] for h in s.hyponyms() for l in h.lemmas()})
    return {"word": word, "synonyms": synonyms, "hypernyms": hypernyms, "hyponyms": hyponyms}
