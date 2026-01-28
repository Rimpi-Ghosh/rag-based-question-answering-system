from sentence_transformers import SentenceTransformer
import numpy as np
from app.ingestion import INDEX, CHUNKS_STORE

# Same model used during ingestion
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(question: str, top_k: int = 3):
    """
    Retrieve top-k relevant chunks from FAISS based on query embedding.
    """
    if INDEX.ntotal == 0:
        return ["No documents have been ingested yet."]

    # Embed the query
    query_embedding = MODEL.encode([question])

    # Search FAISS
    distances, indices = INDEX.search(
        np.array(query_embedding).astype("float32"), top_k
    )

    # Fetch matching chunks
    results = []
    for idx in indices[0]:
        if idx < len(CHUNKS_STORE):
            results.append(CHUNKS_STORE[idx])

    return results
