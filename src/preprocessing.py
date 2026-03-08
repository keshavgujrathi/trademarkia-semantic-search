import os
import re
from typing import Optional, List, Dict

def load_raw_corpus(data_dir: str) -> List[Dict[str, str]]:
    """Enumerates the category subfolders and loads the raw text files."""
    corpus = []
    
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            if os.path.isfile(file_path):
                try:
                    # 20NG dataset requires latin-1, utf-8 will crash
                    with open(file_path, 'r', encoding='latin-1') as f:
                        corpus.append({
                            "text": f.read(),
                            "category": category,
                            "filename": filename
                        })
                except Exception:
                    continue
    
    return corpus

def clean_document(text: str) -> Optional[str]:
    """
    Cleans raw 20NG text. 
    Why: Dense embedding models (like BGE) average out tokens. If we leave 
    in 90s routing paths, PGP blocks, or massive quote chains, the vectors 
    will cluster around "email formatting" instead of the actual topic.
    """
    
    # Strip NNTP headers. The routing path dilutes the actual post content.
    text = re.sub(r'^.*?\n\n', '', text, flags=re.DOTALL)
    
    # Drop email-style blockquotes. We want to embed what the user wrote, 
    # not the entire conversation history, to keep cluster boundaries sharp.
    lines = text.split('\n')
    lines = [line for line in lines if not line.strip().startswith(('>', '|'))]
    text = '\n'.join(lines)
    
    # Nuke PGP signatures. These get broken down into hundreds of garbage 
    # subword tokens by the transformer, completely ruining the embedding.
    text = re.sub(r'-----BEGIN PGP.*?-----END PGP-----', '', text, flags=re.DOTALL)
    
    # Keep only lines with actual alphabet characters (drops ASCII dividers)
    lines = text.split('\n')
    lines = [line for line in lines if line.strip() and re.search(r'[a-zA-Z]', line)]
    text = '\n'.join(lines)
    
    # Strip URLs. They just add noise to topic clustering.
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Drop degraded docs. Anything < 50 chars lacks enough semantic context 
    # to form a meaningful vector anyway.
    if len(text) < 50:
        return None
    
    return text

def build_corpus(data_dir: str) -> List[Dict[str, str]]:
    """Pipeline to load and clean the corpus, returning only viable docs."""
    raw_corpus = load_raw_corpus(data_dir)
    cleaned_corpus = []
    
    for doc in raw_corpus:
        cleaned_text = clean_document(doc["text"])
        if cleaned_text:
            cleaned_corpus.append({
                "text": cleaned_text,
                "category": doc["category"],
                "filename": doc["filename"]
            })
    
    retained = len(cleaned_corpus)
    total = len(raw_corpus)
    
    print(f"Corpus built: {retained}/{total} docs retained ({(retained/total)*100:.1f}%)")
    
    return cleaned_corpus