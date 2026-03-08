# Trademarkia Semantic Search

A semantic search and caching system built on the UCI 20 Newsgroups dataset. The system uses fuzzy clustering with Gaussian Mixture Models to group documents by semantic similarity, maintains a cluster-partitioned semantic cache for query acceleration, and exposes a FastAPI service for real-time search with cache statistics.

**Live Demo:** https://keshav-semantic-search.streamlit.app

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the UCI 20 Newsgroups download at `data/raw` so that subdirectories like `data/raw/sci.space/` exist.

## Building the Index

Runs preprocessing, embedding (BAAI/bge-small-en-v1.5), BIC curve analysis, and GMM fitting in sequence. Takes ~15 minutes on first run.

```bash
python build_index.py
```

## Starting the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Cluster Analysis Dashboard

To review the mathematical justification for the k=30 decision — the BIC 
curve, UMAP projection, cluster term distributions, and boundary case 
entropy analysis:
```bash
streamlit run streamlit_app.py
```

The dashboard also exposes the full search interface with live cache 
hit/miss tracking.

## Endpoints

```bash
# Search with cache
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "space shuttle launches and NASA missions"}' | python3 -m json.tool

# Cache statistics
curl -s http://localhost:8000/cache/stats | python3 -m json.tool

# Clear cache
curl -s -X DELETE http://localhost:8000/cache | python3 -m json.tool

# Inspect cluster bucket
curl -s "http://localhost:8000/cache/inspect?cluster_id=3" | python3 -m json.tool
```

## Docker

```bash
docker build -t trademarkia-semantic-search .
docker run -p 8000:8000 trademarkia-semantic-search
```

## Design Decisions

- **BGE model**: bge-small-en-v1.5 consistently outperforms MiniLM-L6-v2 on the MTEB retrieval benchmark at comparable model size. It uses asymmetric query/document encoding — queries get a "Represent this sentence:" prefix at inference time, documents do not — which better reflects the asymmetry between short queries and long news posts in this corpus.

- **FAISS over ChromaDB**: FAISS is what production retrieval systems actually use — it has no server overhead, no persistence abstraction, and exposes exact control over index type. We build IndexFlatIP (exact inner product) on L2-normalized vectors, which is equivalent to cosine similarity without the division. A thin metadata layer (pickle) handles corpus association.

- **GMM + BIC for clustering**: Hard clustering (KMeans) was explicitly ruled out by the task — a document about gun legislation belongs to both politics and firearms to varying degrees. GMM produces a probability distribution over clusters per document. BIC was used to select k, but the curve was monotonically decreasing through k=49, meaning the 20NG semantic space has no clean elbow — it is dense and continuously overlapping. k=30 was chosen as an operational ceiling: beyond this, cluster sizes become too small for meaningful cache routing and marginal BIC improvement flattens.

- **Cluster-partitioned cache**: The cache is a two-level dict keyed by dominant cluster id. Lookup is restricted to the matching cluster bucket, giving O(n/k) average complexity instead of O(n) over the full cache. This is not just an optimization — it is semantically motivated. Two queries that land in different clusters are unlikely to be asking the same thing, so comparing them is both wasteful and prone to false positives.

- **Similarity threshold at 0.80**: Empirically derived by testing four query pairs at five threshold values (0.70–0.90). At 0.85, a genuine paraphrase pair ("space shuttle launches and NASA missions" vs "NASA rocket programs and space exploration", similarity 0.812) is incorrectly treated as a miss. At 0.75, topic-adjacent but semantically distinct queries (similarity ~0.789) start collapsing incorrectly. 0.80 is the value that correctly separates paraphrases from related-but-distinct queries on BGE embeddings.
