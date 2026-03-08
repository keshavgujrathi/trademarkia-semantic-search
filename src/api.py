from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.cache import SemanticCache
from src.clustering import load_clustering
from src.embeddings import embed_query, load_index, search


# PYDANTIC SCHEMAS
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: str
    dominant_cluster: int

class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float

class CacheClearResponse(BaseModel):
    status: str


# LIFESPAN STATE MANAGEMENT
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads heavy ML models (BGE, FAISS, GMM) into memory exactly once on startup.
    Prevents memory leaks and ensures the API doesn't reload 150MB of assets per request.
    """
    # Load FAISS index and corpus
    app.state.index, _, app.state.corpus = load_index("faiss_index/corpus")
    
    # Load clustering models
    app.state.pca, app.state.gmm, _ = load_clustering("faiss_index/clustering")
    
    # Initialize cache (0.80 threshold for BGE embeddings)
    app.state.cache = SemanticCache(similarity_threshold=0.80)
    
    yield


app = FastAPI(
    title="Trademarkia Semantic Search API",
    lifespan=lifespan
)


# ENDPOINTS
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    state = app.state
    
    # Embed query to trigger BAAI/bge-small-en-v1.5
    query_vec = embed_query(request.query)
    
    # Dimensionality reduction & GMM routing
    query_reduced = state.pca.transform(query_vec.reshape(1, -1))
    soft_assignments = state.gmm.predict_proba(query_reduced)[0]
    dominant_cluster = int(np.argmax(soft_assignments))
    
    # O(N/K) Cache Lookup
    cached_entry, similarity = state.cache.lookup(query_vec, dominant_cluster)
    
    if cached_entry is not None:
        state.cache.record_hit()
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=cached_entry.query,
            similarity_score=similarity,
            result=cached_entry.result,
            dominant_cluster=dominant_cluster
        )
        
    # Cache Miss - Execute Vector Search
    state.cache.record_miss()
    scores, indices = search(state.index, query_vec, k=1)
    best_doc = state.corpus[indices[0]]
    
    # Format the result string (adding metadata directly into the text for context)
    result_str = f"[Category: {best_doc['category']}] | {best_doc['text'][:800]}..."
    
    # Store in cache
    state.cache.store(request.query, query_vec, result_str, dominant_cluster)
    
    return QueryResponse(
        query=request.query,
        cache_hit=False,
        result=result_str,
        dominant_cluster=dominant_cluster
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    return CacheStatsResponse(**app.state.cache.stats())


@app.delete("/cache", response_model=CacheClearResponse)
async def clear_cache():
    app.state.cache.flush()
    return CacheClearResponse(status="cache flushed successfully")