from fastapi import UploadFile
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ------------------ GLOBAL OBJECTS ------------------

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_DIM = 384  # Dimension for MiniLM embeddings
INDEX = faiss.IndexFlatL2(VECTOR_DIM)

CHUNKS_STORE = []  # Keeps text chunks aligned with vectors


# ------------------ Chunking ------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


# ------------------ Ingestion ------------------

def ingest_document(file: UploadFile):
    print("🔥 ingest_document() CALLED")

    text = ""

    # Read document
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        for page in reader.pages:
            text += page.extract_text() or ""

    elif file.filename.endswith(".txt"):
        text = file.file.read().decode("utf-8")

    else:
        raise ValueError("Unsupported file format")

    # Chunk text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = MODEL.encode(chunks)

    # Store in FAISS
    INDEX.add(np.array(embeddings).astype("float32"))
    CHUNKS_STORE.extend(chunks)

    # Logs for verification
    print(f"\nDocument ingested: {file.filename}")
    print(f"Characters: {len(text)}")
    print(f"Chunks added: {len(chunks)}")
    print(f"Total vectors in index: {INDEX.ntotal}")

