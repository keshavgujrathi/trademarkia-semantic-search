import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
import umap
from sklearn.mixture import GaussianMixture

os.makedirs("outputs/figures", exist_ok=True)

with open("faiss_index/corpus.pkl", "rb") as f:
    data = pickle.load(f)
embeddings = data["embeddings"]
corpus = data["corpus"]

with open("faiss_index/clustering.pkl", "rb") as f:
    cl = pickle.load(f)
gmm = cl["gmm"]
soft_assignments = cl["soft_assignments"]
pca = cl["pca"]
bic_scores = None  # curve was monotonic, no scores to plot

# FIG 1: bic_curve.png
reduced_embeddings = pca.transform(embeddings)
k_range = range(8, 50)
bic_curve_scores = []

for k in k_range:
    gmm_temp = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        max_iter=100,
        n_init=1,
        random_state=42
    )
    gmm_temp.fit(reduced_embeddings)
    bic_curve_scores.append(gmm_temp.bic(reduced_embeddings))

plt.figure(figsize=(10, 6))
plt.plot(k_range, bic_curve_scores, 'b-', linewidth=2)
plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='operational cap k=30')
plt.xlabel("k")
plt.ylabel("BIC Score")
plt.title("BIC Score vs Number of Clusters (20NG Corpus)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/figures/bic_curve.png", dpi=150)
plt.close()
print("Saved bic_curve.png")

# FIG 2: umap_clusters.png
umap_reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42, n_neighbors=15, min_dist=0.1)
umap_embeddings = umap_reducer.fit_transform(embeddings)
dominant_clusters = np.argmax(soft_assignments, axis=1)

plt.figure(figsize=(14, 10))
cmap = plt.colormaps.get_cmap("tab20b").resampled(30)
scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                     c=dominant_clusters, cmap=cmap, alpha=0.35, s=2, linewidths=0)
cbar = plt.colorbar(scatter)
cbar.set_label("Cluster ID")
plt.title("UMAP Projection — 30 GMM Clusters (20NG Corpus)")
plt.tight_layout()
plt.savefig("outputs/figures/umap_clusters.png", dpi=150)
plt.close()
print("Saved umap_clusters.png")

texts = [doc["text"] for doc in corpus]
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=2)
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# FIG 3: boundary_cases.png
doc_entropies = np.array([entropy(row) for row in soft_assignments])
top_15_indices = np.argsort(doc_entropies)[-15:][::-1]  # Top 15 highest entropy

fig = plt.figure(figsize=(16, 20))
gs = gridspec.GridSpec(15, 2, figure=fig, width_ratios=[3, 1])

for i, doc_idx in enumerate(top_15_indices):
    doc_text = corpus[doc_idx]["text"]
    category = corpus[doc_idx]["category"]
    truncated_text = doc_text[:140] + "..." if len(doc_text) > 140 else doc_text
    
    # Left col: text
    ax_text = fig.add_subplot(gs[i, 0])
    ax_text.text(0.05, 0.95, f"{truncated_text}\n\nCategory: {category}", 
                 transform=ax_text.transAxes, fontsize=8, wrap=True,
                 verticalalignment="top")
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    
    # Right col: cluster probabilities
    ax_bar = fig.add_subplot(gs[i, 1])
    soft_probs = soft_assignments[doc_idx]
    top_4_indices = np.argsort(soft_probs)[-4:][::-1]
    top_4_probs = soft_probs[top_4_indices]
    top_4_clusters = top_4_indices
    
    y_pos = np.arange(4)
    ax_bar.barh(y_pos, top_4_probs, color="#5B8DB8")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f"Cluster {c}" for c in top_4_clusters], fontsize=7)
    ax_bar.set_xlabel("Probability", fontsize=7)
    ax_bar.tick_params(axis='both', which='major', labelsize=7)

plt.suptitle("Most Uncertain Documents — Boundary Cases", fontsize=12, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/figures/boundary_cases.png", dpi=150)
plt.close()
print("Saved boundary_cases.png")

# BOUNDARY REPORT
with open("outputs/boundary_report.txt", "w") as f:
    for doc_idx in top_15_indices:
        doc = corpus[doc_idx]
        top2 = np.argsort(soft_assignments[doc_idx])[-2:][::-1]
        
        # top 5 TF-IDF terms for each of the top 2 clusters
        cluster0_terms = []
        cluster1_terms = []
        
        for cluster_id in top2:
            cluster_mask = dominant_clusters == cluster_id
            cluster_tfidf = tfidf_matrix[cluster_mask]
            cluster_scores = np.sum(cluster_tfidf, axis=0).A1
            top_5_indices = np.argsort(cluster_scores)[-5:][::-1]
            top_5_terms = feature_names[top_5_indices]
            
            if cluster_id == top2[0]:
                cluster0_terms = top_5_terms
            else:
                cluster1_terms = top_5_terms
        
        f.write("---\n")
        f.write(f"Document: {doc['filename']} | Category: {doc['category']}\n")
        f.write(f"Text: {doc['text'][:300]}\n\n")
        f.write(f"Cluster tension: Cluster {top2[0]} ({soft_assignments[doc_idx, top2[0]]:.3f}) \n")
        f.write(f"                 vs Cluster {top2[1]} ({soft_assignments[doc_idx, top2[1]]:.3f})\n\n")
        f.write(f"Top terms in Cluster {top2[0]}: {', '.join(cluster0_terms)}\n")
        f.write(f"Top terms in Cluster {top2[1]}: {', '.join(cluster1_terms)}\n")
        f.write("---\n\n")

print("Saved boundary_report.txt")

# FIG 4: cluster_terms.png

dominant_clusters = np.argmax(soft_assignments, axis=1)

fig, axes = plt.subplots(6, 5, figsize=(24, 28))
axes = axes.flatten()

for cluster_id in range(30):
    cluster_mask = dominant_clusters == cluster_id
    cluster_tfidf = tfidf_matrix[cluster_mask]
    
    # Sum TF-IDF scores for this cluster
    cluster_scores = np.sum(cluster_tfidf, axis=0).A1
    
    # top 10 terms
    top_10_indices = np.argsort(cluster_scores)[-10:][::-1]
    top_10_scores = cluster_scores[top_10_indices]
    top_10_terms = feature_names[top_10_indices]
    
    # Plot
    ax = axes[cluster_id]
    y_pos = np.arange(10)
    ax.barh(y_pos, top_10_scores, color="#7EB87A")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10_terms, fontsize=7)
    ax.set_xlabel("TF-IDF Score", fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_title(f"Cluster {cluster_id}", fontsize=9)

plt.tight_layout(pad=1.5)
plt.savefig("outputs/figures/cluster_terms.png", dpi=150)
plt.close()
print("Saved cluster_terms.png")

# FIG 5: cluster_representatives.png
fig, axes = plt.subplots(30, 3, figsize=(24, 40))

for cluster_id in range(30):
    # top 3 docs with highest membership in this cluster
    cluster_probs = soft_assignments[:, cluster_id]
    top_3_indices = np.argsort(cluster_probs)[-3:][::-1]
    
    # Row label
    fig.text(0.01, (29.5 - cluster_id) / 30, f"Cluster {cluster_id}", 
             rotation=90, fontsize=9, va='center')
    
    for col, doc_idx in enumerate(top_3_indices):
        ax = axes[cluster_id, col]
    
        category = corpus[doc_idx]["category"]
        text = corpus[doc_idx]["text"]
    
        # Escape any dollar signs to prevent Matplotlib math-mode triggers
        clean_text = text[:180].replace("$", r"\$") + "..." if len(text) > 180 else text.replace("$", r"\$")
        prob = soft_assignments[doc_idx, cluster_id]
    
        ax.text(0.05, 0.95, f"{category}", transform=ax.transAxes, 
                fontsize=8, fontweight='bold', va='top')
        ax.text(0.05, 0.85, clean_text, transform=ax.transAxes, 
                fontsize=7, va='top', wrap=True)
        ax.text(0.95, 0.05, f"p={prob:.3f}", transform=ax.transAxes, 
                fontsize=7, color='gray', ha='right', va='bottom')
    
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Most Representative Documents per Cluster", fontsize=13, y=0.995)
plt.tight_layout(rect=[0.02, 0, 1, 0.993])
plt.savefig("outputs/figures/cluster_representatives.png", dpi=150)
plt.close()
print("Saved cluster_representatives.png")