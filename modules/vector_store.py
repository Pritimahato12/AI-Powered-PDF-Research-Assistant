import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


def create_vector_index(embeddings):
    """Create a cosine-similarity FAISS index."""
    vectors = np.asarray(embeddings, dtype=np.float32).copy()

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("No embeddings available to index.")

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _tokenize(text):
    """Simple lowercase word tokenizer for BM25."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def search_hybrid_chunks(
    question,
    model,
    index,
    chunks,
    top_k=6,
    candidate_k=15
):
    """
    Combine FAISS semantic retrieval and BM25 keyword retrieval
    with Reciprocal Rank Fusion (RRF).
    """
    if not chunks or index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )
    query_vector = np.asarray(query_embedding, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    semantic_k = min(candidate_k, index.ntotal)
    semantic_scores, semantic_indices = index.search(
        query_vector,
        semantic_k
    )

    tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    keyword_scores = bm25.get_scores(_tokenize(question))

    keyword_k = min(candidate_k, len(chunks))
    keyword_indices = np.argsort(keyword_scores)[::-1][:keyword_k]

    # Reciprocal Rank Fusion: robustly combines differently scaled scores.
    rrf_scores = {}
    semantic_score_map = {}
    keyword_score_map = {}

    for rank, (score, idx) in enumerate(
        zip(semantic_scores[0], semantic_indices[0]), start=1
    ):
        if idx < 0:
            continue
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
        semantic_score_map[idx] = float(score)

    for rank, idx in enumerate(keyword_indices, start=1):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
        keyword_score_map[idx] = float(keyword_scores[idx])

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:top_k]

    results = []
    for idx in ranked_indices:
        chunk = chunks[idx]
        results.append({
            "score": float(rrf_scores[idx]),
            "semantic_score": semantic_score_map.get(idx, 0.0),
            "keyword_score": keyword_score_map.get(idx, 0.0),
            "source": chunk.get("source", "Unknown PDF"),
            "page_number": chunk["page_number"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"]
        })

    return results


# Kept for backward compatibility with the earlier project.
def search_similar_chunks(question, model, index, chunks, top_k=4):
    return search_hybrid_chunks(
        question=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )
