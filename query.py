import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from config import GEMINI_API_KEY

# Load FAISS index and metadata
VECTOR_DIR = "vectorstore"
INDEX_FILE = f"{VECTOR_DIR}/index.faiss"
META_FILE = f"{VECTOR_DIR}/metadata.pkl"

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

# Load same embedding model used during ingestion
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
model_name = "models/gemini-2.5-flash"

def search_faiss(query, top_k=5):
    query_emb = embedder.encode([query])
    distances, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        meta = metadata[idx]
        results.append({
            "text": None,  # We'll load text only when needed
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"]
        })
    return indices[0], distances[0]

def load_texts_for_indices(indices):
    # Reconstruct text chunks from ingestion step
    # We didn’t save texts to disk in ingest.py, so let’s quickly re-read them
    import pdfplumber, os
    data_dir = "data"
    docs = {}
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            docs[file] = []
            with pdfplumber.open(os.path.join(data_dir, file)) as pdf:
                for page in pdf.pages:
                    docs[file].append(page.extract_text() or "")
    return docs

def generate_answer(query):
    print(f"Searching for: {query}")
    indices, distances = search_faiss(query)
    docs = load_texts_for_indices(indices)

    # Collect retrieved chunks
    retrieved_texts = []
    for idx in indices:
        meta = metadata[idx]
        src = meta["source"]
        page = meta["page"]
        text = docs[src][page - 1]
        retrieved_texts.append(f"From {src} (page {page}):\n{text}")

    # Combine into a context prompt
    context = "\n\n".join(retrieved_texts[:3])  # use top 3 pages
    prompt = f"""
You are a helpful research assistant. 
Use the following context from research papers to answer the question.

Context:
{context}

Question: {query}

Answer in concise academic style:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    print("\nAnswer:")
    print(response.text)

if __name__ == "__main__":
    user_query = input("Enter your research question: ")
    generate_answer(user_query)
