import os
import streamlit as st
import numpy as np
from src.embeddings import embed_query, load_index, search
from src.clustering import load_clustering
from src.cache import SemanticCache

st.set_page_config(page_title="Trademarkia Semantic Search", layout="wide")

# Load resources
@st.cache_resource
def load_resources():
    index, embeddings, corpus = load_index("faiss_index/corpus")
    pca, gmm, soft_assignments = load_clustering("faiss_index/clustering")
    cache = SemanticCache(similarity_threshold=0.80)
    return index, embeddings, corpus, pca, gmm, soft_assignments, cache

index, embeddings, corpus, pca, gmm, soft_assignments, cache = load_resources()

# SIDEBAR STATS
with st.sidebar:
    st.subheader("⚙️ System Telemetry")
    stats = cache.stats()
    st.metric("Total Cache Entries", stats["total_entries"])
    st.metric("Hit Rate", f"{stats['hit_rate']*100:.1f}%")
    st.write(f"Hits: {stats['hit_count']} | Misses: {stats['miss_count']}")
    
    if st.button("Clear Cache"):
        cache.flush()
        st.success("Cache cleared!")
        st.rerun()

# Instantiate tabs
tab1, tab2 = st.tabs(["🔍 Search Engine", "📊 Cluster Analysis"])

# TAB 1: Search
with tab1:
    st.title("Trademarkia Semantic Search")
    st.write("An $O(N/K)$ cluster-routed search engine with a vectorized semantic cache.")
    
    query = st.text_input("Enter a query (Press Enter to search):", placeholder="e.g., NASA space shuttle missions")
    
    if query:
        # Embed & Route
        query_vec = embed_query(query)
        query_reduced = pca.transform(query_vec.reshape(1, -1))
        soft_assignments_query = gmm.predict_proba(query_reduced)[0]
        dominant_cluster = int(np.argmax(soft_assignments_query))
        
        # Cache Lookup
        cached_entry, similarity = cache.lookup(query_vec, dominant_cluster)
        
        with st.container(border=True):
            if cached_entry is not None:
                cache.record_hit()
                st.success("🎯 CACHE HIT")
                st.write(f"**Matched Query:** `{cached_entry.query}` (Similarity: {similarity:.3f})")
                st.write(f"**Dominant Cluster:** {dominant_cluster}")
                st.markdown("---")
                # cache stores the flat string directly
                st.text_area("Result", value=cached_entry.result, height=200)
                
            else:
                cache.record_miss()
                st.warning("⚡ CACHE MISS — Performing Vector Search")
                
                # FAISS search
                scores, indices = search(index, query_vec, k=1)
                best_doc = corpus[indices[0]]
                
                # Format string to match API schema
                result_str = f"[Category: {best_doc['category']}] | {best_doc['text'][:800]}..."
                
                # Store strictly the string in cache
                cache.store(query, query_vec, result_str, dominant_cluster)
                
                st.write(f"**Dominant Cluster:** {dominant_cluster}")
                st.markdown("---")
                st.text_area("Result", value=result_str, height=200)

# TAB 2: Cluster Analysis
with tab2:
    st.title("Model Audit & Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("BIC Curve Analysis")
        if os.path.exists("outputs/figures/bic_curve.png"):
            st.image("outputs/figures/bic_curve.png", caption="Monotonic curve indicating no clean elbow. Operationally capped at k=30 for cache density.")
            
        st.subheader("Cluster TF-IDF Terms")
        if os.path.exists("outputs/figures/cluster_terms.png"):
            st.image("outputs/figures/cluster_terms.png", caption="Top keywords defining semantic boundaries.")

    with col2:
        st.subheader("UMAP 2D Projection")
        if os.path.exists("outputs/figures/umap_clusters.png"):
            st.image("outputs/figures/umap_clusters.png", caption="384-dim BGE embeddings reduced to 2D.")
            
        st.subheader("Boundary Case Entropy")
        if os.path.exists("outputs/figures/boundary_cases.png"):
            st.image("outputs/figures/boundary_cases.png", caption="Documents with maximum GMM uncertainty (Shannon Entropy).")

    if os.path.exists("outputs/boundary_report.txt"):
        with st.expander("📄 View Detailed Boundary Case Report"):
            with open("outputs/boundary_report.txt", "r") as f:
                st.text(f.read())