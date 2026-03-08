import pickle
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder() -> SentenceTransformer:
    """
    Loads and caches the BGE embedding model.
    Why BGE-small: It's extremely lightweight (384-dim, ~33M params) so it runs locally, 
    but it consistently beats older models like MiniLM on the MTEB leaderboard. 
    It also supports asymmetric retrieval, which is ideal for matching short queries 
    to long Usenet documents.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder

def embed_documents(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Generates L2-normalized document embeddings."""
    embedder = get_embedder()
    
    # encode() natively handles batching, TQDM, and the L2 normalization 
    # required for FAISS Inner Product search.
    embeddings = embedder.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings.astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    """Embeds a search query with the required BGE instruction prefix."""
    embedder = get_embedder()
    
    # Asymmetric mapping: BGE requires this exact prefix for short queries 
    # to align them properly with the un-prefixed document embeddings.
    query_with_prefix = f"Represent this sentence for searching relevant passages: {query}"
    
    embedding = embedder.encode(
        query_with_prefix, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embedding.astype(np.float32)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds an exact-search Inner Product index (equivalent to Cosine on L2-normalized vectors).
    Why IndexFlatIP: For a small corpus of ~20,000 documents, exact search (Flat) is 
    computationally trivial (sub-millisecond). Using an ANN algorithm like HNSW or IVF 
    here would just waste memory and require tuning, with zero noticeable speedup.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def save_index(index: faiss.Index, embeddings: np.ndarray, corpus: List[dict], path: str) -> None:
    """Persists the FAISS index and the raw corpus metadata."""
    faiss.write_index(index, f"{path}.index")
    
    with open(f"{path}.pkl", "wb") as f:
        pickle.dump({"corpus": corpus, "embeddings": embeddings}, f)

def load_index(path: str) -> Tuple[faiss.Index, np.ndarray, List[dict]]:
    index = faiss.read_index(f"{path}.index")
    
    with open(f"{path}.pkl", "rb") as f:
        data = pickle.load(f)
        
    return index, data["embeddings"], data["corpus"]

def search(index: faiss.Index, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    query_vec = query_vec.reshape(1, -1)
    scores, indices = index.search(query_vec, k)
    return scores[0], indices[0]