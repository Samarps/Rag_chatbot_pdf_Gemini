import os
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Paths
DATA_DIR = "data"
VECTOR_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
META_FILE = os.path.join(VECTOR_DIR, "metadata.pkl")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper: read all PDFs and extract text
def read_pdfs(data_dir):
    docs = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            path = os.path.join(data_dir, file)
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        docs.append({"text": text, "source": file, "page": i + 1})
    return docs

# Helper: chunk text
def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent.split())
        if current_length + sent_len > max_length:
            chunks.append(". ".join(current_chunk))
            current_chunk, current_length = [sent], sent_len
        else:
            current_chunk.append(sent)
            current_length += sent_len
    if current_chunk:
        chunks.append(". ".join(current_chunk))
    return chunks

# Read and chunk
print("Reading PDFs...")
docs = read_pdfs(DATA_DIR)

texts, metadata = [], []
for doc in docs:
    chunks = chunk_text(doc["text"])
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "source": doc["source"],
            "page": doc["page"],
            "chunk_id": i
        })

# Create embeddings
print("Creating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index and metadata
os.makedirs(VECTOR_DIR, exist_ok=True)
faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "wb") as f:
    pickle.dump(metadata, f)

print(f"Ingestion complete! {len(texts)} chunks stored in FAISS.")
