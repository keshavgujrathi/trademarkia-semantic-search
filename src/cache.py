import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: dict
    dominant_cluster: int
    timestamp: float


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.buckets: Dict[int, List[CacheEntry]] = {}
        
        # THE TUNABLE DECISION:
        # If threshold -> 0.99: Acts like a dumb exact-match cache (useless).
        # If threshold -> 0.70: Causes "semantic drift" (e.g., "space launch" matches "satellite repair").
        # 0.85 is used for BGE embeddings where user intent is preserved but exact phrasing can vary.
        self.similarity_threshold = similarity_threshold
        
        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding: np.ndarray, dominant_cluster: int) -> Tuple[Optional[CacheEntry], Optional[float]]:
        """
        Why buckets: A flat cache is O(N) to search. By routing the query to its GMM 
        dominant cluster bucket first, we drop the search space massively. It becomes an 
        O(N/K) operation, which scales easily.
        """
        cluster_id = int(dominant_cluster)
        bucket = self.buckets.get(cluster_id, [])
        
        if not bucket:
            return None, None
        
        # Vectorized similarity computation
        query_vec = query_embedding.flatten()
        bucket_embeddings = np.stack([entry.embedding for entry in bucket])
        similarities = np.dot(bucket_embeddings, query_vec)
        
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_entry = bucket[best_idx]

        if best_score >= self.similarity_threshold:
            return best_entry, best_score
            
        return None, None

    def store(self, query: str, embedding: np.ndarray, result: dict, dominant_cluster: int) -> None:
        cluster_id = int(dominant_cluster)
        
        if cluster_id not in self.buckets:
            self.buckets[cluster_id] = []
        
        entry = CacheEntry(
            query=query,
            embedding=embedding.flatten(),
            result=result,
            dominant_cluster=cluster_id,
            timestamp=time.time()
        )
        
        self.buckets[cluster_id].append(entry)

    def record_hit(self) -> None:
        self.hit_count += 1

    def record_miss(self) -> None:
        self.miss_count += 1

    def stats(self) -> dict:
        total_entries = sum(len(bucket) for bucket in self.buckets.values())
        total_queries = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_queries if total_queries > 0 else 0.0
        
        return {
            "total_entries": total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def flush(self) -> None:
        self.buckets.clear()
        self.hit_count = 0
        self.miss_count = 0

    def get_bucket(self, cluster_id: int) -> List[CacheEntry]:
        return self.buckets.get(int(cluster_id), [])