import numpy as np
from src.embeddings import embed_query

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

pairs = [
    # near-identical phrasing
    ("space shuttle launches and NASA missions",
     "space shuttle launches and NASA programs"),
    # paraphrase
    ("space shuttle launches and NASA missions",
     "NASA rocket programs and space exploration"),
    # same topic, different angle
    ("gun control legislation in America",
     "second amendment rights and firearms policy"),
    # related but distinct
    ("Middle East conflict and Israeli politics",
     "Palestinian territories and Arab relations"),
]

# Embed all queries and compute similarities
results = []
for q1, q2 in pairs:
    emb1 = embed_query(q1)
    emb2 = embed_query(q2)
    similarity = float(np.dot(emb1, emb2))
    results.append((q1, q2, similarity))

# Print table header
print("Threshold Analysis Results")
print("=" * 80)
print(f"{'Query 1 (first 35 chars)':<35} |", end="")
for thresh in thresholds:
    print(f" {thresh:<6} |", end="")
print()
print("-" * 80)

# Print table rows
for q1, q2, similarity in results:
    q1_short = q1[:35] + "..." if len(q1) > 35 else q1
    print(f"{q1_short:<35} |", end="")
    
    for thresh in thresholds:
        hit_miss = "HIT" if similarity >= thresh else "miss"
        print(f" {hit_miss:<6} |", end="")
    print()
    
    # Second line with similarity scores
    print(f"{'':35} |", end="")
    for thresh in thresholds:
        print(f" {similarity:.3f}  |", end="")
    print()
    print()

# Interpretation
print("Interpretation:")
print("=" * 80)

for thresh in thresholds:
    hits = [similarity for _, _, similarity in results if similarity >= thresh]
    hit_count = len(hits)
    
    if thresh == 0.70:
        interpretation = f"0.70 allows all semantic similarities including paraphrases and related topics, but may include less relevant matches."
    elif thresh == 0.75:
        interpretation = f"0.75 captures paraphrases and strong semantic relationships while filtering out some tangentially related queries."
    elif thresh == 0.80:
        interpretation = f"0.80 focuses on paraphrases and very similar phrasing, missing broader topic-level similarities."
    elif thresh == 0.85:
        interpretation = f"0.85 only matches near-identical phrasing and very close paraphrases, missing most semantic variations."
    else:  # 0.90
        interpretation = f"0.90 is extremely restrictive, matching only virtually identical queries with minimal variation."
    
    print(f"Threshold {thresh}: {hit_count}/4 hits. {interpretation}")
